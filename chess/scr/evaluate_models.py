"""Compare two frozen models with equal-budget MCTS and a fixed value benchmark.

This is a small paired diagnostic, not an Elo estimate. Unfinished benchmark
games have unknown outcomes and do not enter value metrics. This evaluation
rule does not change the training rule that labels move-limit games as draws.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

try:
    from .chess_engine import GameState
    from .evaluate_human import OPENINGS, RESULT_VALUES, sha256_file
    from .experiment import utc_now, write_json
    from .mcts import best_move, mcts_search
    from .rl_interface import encode_gamestate, load_model
    from .self_play import terminal_details
except ImportError:
    from chess_engine import GameState
    from evaluate_human import OPENINGS, RESULT_VALUES, sha256_file
    from experiment import utc_now, write_json
    from mcts import best_move, mcts_search
    from rl_interface import encode_gamestate, load_model
    from self_play import terminal_details


def build_openings(match_pairs=10, seed=1):
    """Six short named openings, then seeded legal two-ply continuations.

    The extra continuations are a reproducible coverage fixture, not curated
    opening theory. The exact histories are frozen in each comparison output.
    """
    if not 0 <= match_pairs <= 20:
        raise ValueError("match_pairs must be between 0 and 20")
    rng, openings, seen = random.Random(seed), [], set()
    for index in range(match_pairs):
        name, base = OPENINGS[index % len(OPENINGS)]
        for _ in range(1000):
            state, moves = GameState(), list(base)
            for uci in moves:
                state.push_uci(uci)
            if index >= len(OPENINGS):
                for _ in range(2):
                    legal = sorted(state.get_all_legal_moves(state.player, as_moves=True), key=lambda m: m.uci())
                    if terminal_details(state) is not None or not legal:
                        break
                    move = rng.choice(legal)
                    moves.append(move.uci())
                    state.make_move(move)
            identity = tuple(moves)
            if identity not in seen and terminal_details(state) is None:
                seen.add(identity)
                openings.append({"pair": index + 1,
                                 "name": name if index < len(OPENINGS) else f"seeded_{name}_{index + 1}",
                                 "moves": moves, "fen_after_opening": state.to_fen()})
                break
        else:
            raise RuntimeError("Could not construct distinct playable openings")
    return openings


def load_benchmark(directory):
    """Read only a completed benchmark whose game-record hash still matches."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Value benchmark is not complete")
    game_bytes = (directory / "games.jsonl").read_bytes()
    if hashlib.sha256(game_bytes).hexdigest() != manifest.get("games_sha256"):
        raise ValueError("Value benchmark games failed their SHA256 integrity check")
    records, seen = [], set()
    canonical = manifest.get("kind") == "frozen_selfplay_value_benchmark"
    for line in game_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        game_id = record.get("game_id")
        if not isinstance(game_id, str) or not game_id or game_id in seen:
            raise ValueError("Benchmark game_id must be a unique nonempty string")
        seen.add(game_id)
        if record.get("result") not in {*RESULT_VALUES, "*"}:
            raise ValueError(f"Unknown outcome in benchmark game {game_id}")
        if not isinstance(record.get("truncated"), bool) or not isinstance(record.get("termination"), str):
            raise ValueError(f"Missing termination/truncated metadata in benchmark game {game_id}")
        if not isinstance(record.get("moves"), list) or not all(isinstance(m, str) for m in record["moves"]):
            raise ValueError(f"Invalid move history in benchmark game {game_id}")
        if not isinstance(record.get("initial_fen"), str):
            raise ValueError(f"Missing initial FEN in benchmark game {game_id}")
        if canonical and (record["truncated"] != (record["result"] == "*") or
                          record["truncated"] != (record["termination"] == "move_limit")):
            raise ValueError(f"Inconsistent cutoff/result metadata in benchmark game {game_id}")
        if "plies" in record and record["plies"] != len(record["moves"]):
            raise ValueError(f"Ply count mismatch in benchmark game {game_id}")
        records.append(record)
    if not records:
        raise ValueError("Value benchmark contains no games")
    if canonical and manifest.get("counts", {}).get("games") != len(records):
        raise ValueError("Value benchmark manifest game count does not match its records")
    return records, manifest


def _replay_game(game):
    """Validate all histories, even excluded cutoffs, retaining repetition state."""
    state, positions = GameState.from_fen(game["initial_fen"]), []
    excluded = game["truncated"] or game["result"] == "*" or game["termination"] == "move_limit"
    for ply, uci in enumerate(game["moves"], 1):
        if terminal_details(state) is not None:
            raise ValueError(f"Move after terminal position in benchmark game {game['game_id']} at ply {ply}")
        move = state.parse_uci(uci)
        if not excluded:
            positions.append((state.fast_copy_for_search(), uci))
        state.make_move(move)
    if game.get("final_fen") is not None and game["final_fen"] != state.to_fen():
        raise ValueError(f"Final FEN mismatch in benchmark game {game['game_id']}")
    ending = terminal_details(state)
    if excluded and ending is not None:
        raise ValueError(f"Unknown/cutoff outcome has a terminal final board in benchmark game {game['game_id']}")
    if not excluded:
        if ending is None or ending != (game["termination"], RESULT_VALUES[game["result"]]):
            raise ValueError(f"Completed outcome/termination mismatch in benchmark game {game['game_id']}")
    return positions, excluded


@torch.inference_mode()
def _values(model, encoded):
    device = next(model.parameters()).device
    _, values = model(encoded.to(device))
    values = values.reshape(-1).cpu().tolist()
    if len(values) != len(encoded) or not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite or incorrectly shaped value predictions")
    return values


def _value_metrics(rows):
    count = len(rows)
    result = {"positions": count, "games": len({row["game_id"] for row in rows})}
    for name in ("baseline", "candidate", "zero"):
        result[name] = {"mse": sum(row[f"{name}_squared_error"] for row in rows) / count if count else None,
                        "mae": sum(row[f"{name}_absolute_error"] for row in rows) / count if count else None,
                        "mean_prediction": sum(row.get(f"{name}_value", 0.) for row in rows) / count if count else None,
                        "mean_absolute_prediction": sum(abs(row.get(f"{name}_value", 0.)) for row in rows) / count if count else None}
    for metric in ("mse", "mae"):
        result[f"candidate_minus_baseline_{metric}"] = (result["candidate"][metric] - result["baseline"][metric]
                                                         if count else None)
    return result


def _quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def bootstrap_games(games, samples=2000, seed=1):
    """Resample whole games as paired clusters, retaining position weighting."""
    eligible = [game for game in games if game["positions_scored"]]
    result = {"unit": "whole game, paired between models", "samples": samples, "seed": seed,
              "confidence_level": .95, "scored_games": len(eligible),
              "interpretation": "descriptive percentile interval on this fixed benchmark; not a strength confidence interval",
              "candidate_minus_baseline_mse": None, "candidate_minus_baseline_mae": None}
    if len(eligible) < 2 or samples == 0:
        return result
    rng, differences = random.Random(seed), {"mse": [], "mae": []}
    for _ in range(samples):
        selection = [rng.choice(eligible) for _ in eligible]
        positions = sum(game["positions_scored"] for game in selection)
        for metric in differences:
            differences[metric].append(sum(game["metrics"][f"candidate_minus_baseline_{metric}"] *
                                           game["positions_scored"] for game in selection) / positions)
    for metric, values in differences.items():
        result[f"candidate_minus_baseline_{metric}"] = [_quantile(values, .025), _quantile(values, .975)]
    return result


def score_value_games(baseline, candidate, games, *, batch_size=128, bootstrap_samples=2000,
                     seed=1, position_sink=None, game_sink=None):
    """Both networks see exactly the same encoded pre-move positions, once each."""
    if batch_size < 1 or bootstrap_samples < 0:
        raise ValueError("Require positive batch_size and nonnegative bootstrap_samples")
    baseline.eval()
    candidate.eval()
    all_rows, game_rows = [], []
    for game in games:
        positions, excluded = _replay_game(game)
        rows = []
        for start in range(0, len(positions), batch_size):
            batch = positions[start:start + batch_size]
            encoded = torch.stack([encode_gamestate(state) for state, _ in batch])
            predictions = (_values(baseline, encoded), _values(candidate, encoded))
            for offset, (state, uci) in enumerate(batch):
                ply = start + offset + 1
                target = RESULT_VALUES[game["result"]] * (1 if state.player == "w" else -1)
                row = {"game_id": game["game_id"], "ply": ply, "fen": state.to_fen(),
                       "move": uci, "player": state.player, "result": game["result"],
                       "target_value": target, "phase": "opening" if ply <= 20 else "later",
                       "target_outcome": "win" if target > 0 else "loss" if target < 0 else "draw",
                       "baseline_value": predictions[0][offset], "candidate_value": predictions[1][offset]}
                for name, value in (("baseline", row["baseline_value"]), ("candidate", row["candidate_value"]), ("zero", 0.)):
                    row[f"{name}_squared_error"] = (value - target) ** 2
                    row[f"{name}_absolute_error"] = abs(value - target)
                rows.append(row)
                if position_sink:
                    position_sink(row)
        all_rows.extend(rows)
        game_row = {"game_id": game["game_id"], "result": game["result"], "termination": game["termination"],
                    "truncated": game["truncated"], "excluded_unknown_outcome": excluded,
                    "positions_in_game": len(game["moves"]), "positions_scored": len(rows),
                    "metrics": _value_metrics(rows)}
        game_rows.append(game_row)
        if game_sink:
            game_sink(game_row)
    summary = {"games": len(games), "completed_games": sum(not row["excluded_unknown_outcome"] for row in game_rows),
               "excluded_unknown_outcome_games": sum(row["excluded_unknown_outcome"] for row in game_rows),
               "excluded_unknown_outcome_positions": sum(row["positions_in_game"] for row in game_rows if row["excluded_unknown_outcome"]),
               "termination_counts": dict(Counter(game["termination"] for game in games)),
               "result_counts": dict(Counter(game["result"] for game in games)),
               "value_perspective": "player to move", "aggregation": "sample-weighted over every pre-move position of completed games",
               "target_interpretation": "Observed outcomes under the frozen baseline stochastic benchmark policy; not perfect-play labels",
               "all": _value_metrics(all_rows),
               "by_phase": {phase: _value_metrics([row for row in all_rows if row["phase"] == phase]) for phase in ("opening", "later")},
               "by_target_outcome": {outcome: _value_metrics([row for row in all_rows if row["target_outcome"] == outcome]) for outcome in ("win", "draw", "loss")},
               "by_game_result": {result: _value_metrics([row for row in all_rows if row["result"] == result]) for result in RESULT_VALUES},
               "by_game_outcome": {"decisive": _value_metrics([row for row in all_rows if row["result"] != "1/2-1/2"]),
                                   "draw": _value_metrics([row for row in all_rows if row["result"] == "1/2-1/2"])},
               "bootstrap_game_level": bootstrap_games(game_rows, bootstrap_samples, seed)}
    scored_games = [game for game in game_rows if game["positions_scored"]]
    summary["equal_game_mean_differences"] = {
        metric: sum(game["metrics"][f"candidate_minus_baseline_{metric}"] for game in scored_games) / len(scored_games)
        if scored_games else None for metric in ("mse", "mae")}
    return summary, game_rows


def compare_values(baseline_model, candidate_model, benchmark_dir, *, batch_size=128,
                   bootstrap_samples=2000, seed=1, position_sink=None, game_sink=None):
    """Verified benchmark API for diagnostics across fixed training checkpoints.

    Returns the value summary; optional sinks receive per-position/per-game
    paired records. The caller chooses the models, thread budget, and output.
    This function never selects checkpoints or updates their weights.
    """
    benchmark_dir = Path(benchmark_dir)
    games, manifest = load_benchmark(benchmark_dir)
    hashes = {"games": sha256_file(benchmark_dir / "games.jsonl"),
              "manifest": sha256_file(benchmark_dir / "manifest.json")}
    baseline_model.eval()
    candidate_model.eval()
    summary, _ = score_value_games(baseline_model, candidate_model, games, batch_size=batch_size,
                                   bootstrap_samples=bootstrap_samples, seed=seed,
                                   position_sink=position_sink, game_sink=game_sink)
    if hashes != {"games": sha256_file(benchmark_dir / "games.jsonl"),
                  "manifest": sha256_file(benchmark_dir / "manifest.json")}:
        raise ValueError("Value benchmark changed during comparison")
    summary.update(benchmark_dir=str(benchmark_dir.resolve()), benchmark_sha256=hashes,
                   benchmark_generator_checkpoint_sha256=manifest.get("checkpoint_sha256"),
                   target_interpretation="Observed outcomes under the frozen baseline stochastic benchmark policy; not perfect-play labels")
    return summary


def search_decision(model, state, sims):
    started = time.perf_counter()
    distribution, root = mcts_search(state, model, num_sims=sims, c_puct=1.5,
                                     add_root_dirichlet=False, root=None,
                                     return_root=True, batch_size=16)
    move = best_move(distribution, root)
    if move is None:
        raise ValueError("MCTS returned no move in a nonterminal match position")
    return move, time.perf_counter() - started


def play_match(baseline, candidate, opening, candidate_color, sims=64, max_plies=400):
    if candidate_color not in ("w", "b") or sims < 1 or max_plies < 1:
        raise ValueError("Invalid match color or search/ply budget")
    state, initial_fen = GameState(), GameState().to_fen()
    for uci in opening["moves"]:
        if terminal_details(state) is not None:
            raise ValueError("Opening continues after terminal position")
        state.push_uci(uci)
    search_start_fen = state.to_fen()
    if terminal_details(state) is not None:
        raise ValueError("Opening must end in a playable position")
    started, decisions = time.perf_counter(), []
    seconds, counts = {"baseline": 0., "candidate": 0.}, {"baseline": 0, "candidate": 0}
    ending = None
    for _ in range(max_plies):
        name = "candidate" if state.player == candidate_color else "baseline"
        model = candidate if name == "candidate" else baseline
        move, elapsed = search_decision(model, state, sims)
        decisions.append({"model": name, "player": state.player, "move": move.uci(), "seconds": elapsed})
        seconds[name] += elapsed
        counts[name] += 1
        state.make_move(move)
        ending = terminal_details(state)
        if ending is not None:
            break
    reason, result = ending if ending is not None else ("move_limit", None)
    outcome = "*" if result is None else "1-0" if result > 0 else "0-1" if result < 0 else "1/2-1/2"
    score = None if result is None else .5 if result == 0 else float((result > 0) == (candidate_color == "w"))
    moves_after = [decision["move"] for decision in decisions]
    return {"game_id": f"pair-{opening['pair']}-candidate-{candidate_color}", "pair": opening["pair"],
            "opening": opening["name"], "opening_moves": opening["moves"], "initial_fen": initial_fen,
            "search_start_fen": search_start_fen, "candidate_color": candidate_color,
            "baseline_color": "b" if candidate_color == "w" else "w",
            "moves": list(opening["moves"]) + moves_after, "moves_after_opening": moves_after,
            "decisions": decisions, "plies_after_opening": len(decisions),
            "total_plies": len(opening["moves"]) + len(decisions), "max_plies_after_opening": max_plies,
            "outcome": outcome, "termination": reason, "truncated": ending is None,
            "candidate_score": score, "final_fen": state.to_fen(), "search_seconds": seconds,
            "search_decisions": counts, "duration_seconds": time.perf_counter() - started,
            "search_simulations_per_move": sims}


def match_summary(matches):
    completed = [game for game in matches if game["candidate_score"] is not None and not game.get("truncated")]
    scored_points = sum(game["candidate_score"] for game in completed)
    unresolved = len(matches) - len(completed)
    return {"games": len(matches), "completed_games": len(completed), "unresolved_cutoffs": len(matches) - len(completed),
            "candidate_wins": sum(game["candidate_score"] == 1 for game in completed),
            "baseline_wins": sum(game["candidate_score"] == 0 for game in completed),
            "actual_draws": sum(game["candidate_score"] == .5 for game in completed),
            "candidate_score_on_completed_games": scored_points / len(completed) if completed else None,
            "candidate_score_all_games_bounds": [scored_points / len(matches), (scored_points + unresolved) / len(matches)] if matches else None,
            "bounds_interpretation": "all-game score extremes if every unresolved game became a loss or a win; cutoffs are not assigned draw scores",
            "termination_counts": dict(Counter(game["termination"] for game in matches))}


def compare_models(baseline_checkpoint, candidate_checkpoint, benchmark_dir, output_dir, *,
                   match_pairs=10, sims=64, max_plies=400, threads=4, seed=1,
                   batch_size=128, bootstrap_samples=2000):
    if min(sims, max_plies, threads, batch_size) < 1 or bootstrap_samples < 0:
        raise ValueError("Require positive search/batch budgets and nonnegative bootstrap_samples")
    baseline_checkpoint, candidate_checkpoint, benchmark_dir, output_dir = (
        Path(path).resolve() for path in (baseline_checkpoint, candidate_checkpoint, benchmark_dir, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty comparison directory: {output_dir}")
    openings = build_openings(match_pairs, seed)
    games, manifest = load_benchmark(benchmark_dir)
    hashes = {"baseline_checkpoint": sha256_file(baseline_checkpoint), "candidate_checkpoint": sha256_file(candidate_checkpoint),
              "benchmark_manifest": sha256_file(benchmark_dir / "manifest.json"), "benchmark_games": sha256_file(benchmark_dir / "games.jsonl")}
    config = {"baseline_checkpoint": str(baseline_checkpoint), "candidate_checkpoint": str(candidate_checkpoint),
              "benchmark_dir": str(benchmark_dir), "input_sha256": hashes, "match_pairs": match_pairs,
              "sims": sims, "max_plies_after_opening": max_plies, "threads": threads, "device": "cpu", "seed": seed,
              "batch_size": batch_size, "bootstrap_samples": bootstrap_samples, "root_noise": False,
              "move_sampling": False, "search_tree_reuse": False, "c_puct": 1.5, "search_batch_size": 16,
              "search_budget": "same simulations per decision for both models; no time limit", "started_at": utc_now(),
              "opening_method": "six named two-ply openings then seeded legal two-ply continuations; not curated opening theory",
              "openings_sha256": hashlib.sha256(json.dumps(openings, sort_keys=True).encode()).hexdigest(),
              "source_sha256": {path.name: sha256_file(path) for path in Path(__file__).parent.glob("*.py")}}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "openings.json", openings)
    write_json(output_dir / "benchmark_manifest.json", manifest)
    summary = {"schema_version": 1, "status": "running", "config": config, "limitations": [
        "Both models use equal MCTS simulations with deterministic decisions, fresh roots, and no exploration noise.",
        "A small fixed opening suite is a paired diagnostic, not an Elo estimate or a general strength guarantee.",
        "Unresolved match cutoffs are excluded from the score denominator and are reported separately.",
        "Value labels describe outcomes under the benchmark generation policy, not perfect play or either model's own rollout policy.",
        "Value evaluation excludes unknown cutoff outcomes; training draw-label behavior is unchanged.",
        "The descriptive value interval resamples whole games as paired clusters; positions within a game are not independent.",
    ]}
    write_json(output_dir / "summary.json", summary)
    started, matches, previous_threads = time.perf_counter(), [], torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            baseline, candidate = load_model(baseline_checkpoint, device="cpu"), load_model(candidate_checkpoint, device="cpu")
            with (output_dir / "value_positions.jsonl").open("w") as positions, (output_dir / "value_games.jsonl").open("w") as values:
                summary["values"], _ = score_value_games(
                    baseline, candidate, games, batch_size=batch_size, bootstrap_samples=bootstrap_samples, seed=seed,
                    position_sink=lambda row: positions.write(json.dumps(row, allow_nan=False) + "\n"),
                    game_sink=lambda row: values.write(json.dumps(row, allow_nan=False) + "\n"))
            summary["values"].update(benchmark_dir=str(benchmark_dir),
                                     benchmark_sha256={"games": hashes["benchmark_games"], "manifest": hashes["benchmark_manifest"]},
                                     benchmark_generator_checkpoint_sha256=manifest.get("checkpoint_sha256"))
            print(f"Value benchmark: {summary['values']['all']['positions']} positions; "
                  f"{summary['values']['excluded_unknown_outcome_games']} unresolved games excluded", flush=True)
            write_json(output_dir / "summary.json", summary)
            with (output_dir / "matches.jsonl").open("w") as stream:
                for opening in openings:
                    for color in ("w", "b"):
                        game = play_match(baseline, candidate, opening, color, sims, max_plies)
                        matches.append(game)
                        stream.write(json.dumps(game, allow_nan=False) + "\n")
                        stream.flush()
                        print(f"Match {len(matches)}/{2 * match_pairs}: candidate {color}; {game['termination']}; {game['outcome']}", flush=True)
            current_hashes = {"baseline_checkpoint": sha256_file(baseline_checkpoint), "candidate_checkpoint": sha256_file(candidate_checkpoint),
                              "benchmark_manifest": sha256_file(benchmark_dir / "manifest.json"), "benchmark_games": sha256_file(benchmark_dir / "games.jsonl")}
            if current_hashes != hashes:
                raise ValueError("An evaluation input changed during comparison; results are not reproducible")
            summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        torch.set_num_threads(previous_threads)
        summary.update(matches=match_summary(matches), duration_seconds=time.perf_counter() - started, updated_at=utc_now())
        write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline_checkpoint", "candidate_checkpoint", "benchmark_dir", "output_dir"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--match-pairs", type=int, default=10)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=400, help="Search-played plies after each fixed opening")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    result = compare_models(**vars(parser.parse_args(argv)))
    print(json.dumps({"status": result["status"], "values": result["values"]["all"], "matches": result["matches"]}, indent=2))


if __name__ == "__main__":
    main()
