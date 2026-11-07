"""Compare one human-trained checkpoint with and without MCTS.

Held-out move agreement measures imitation, not move quality. The small tactical
suite and color-paired matches are diagnostics, not an Elo estimate. No examples
from this module are inserted into training data.
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
    from .diagnostics import evaluate_probes
    from .experiment import write_json, utc_now
    from .mcts import best_move, mcts_search
    from .rl_interface import encode_gamestate, load_model, move_to_action
    from .self_play import terminal_details
except ImportError:
    from chess_engine import GameState
    from diagnostics import evaluate_probes
    from experiment import write_json, utc_now
    from mcts import best_move, mcts_search
    from rl_interface import encode_gamestate, load_model, move_to_action
    from self_play import terminal_details


OPENINGS = (
    ("open_game", ("e2e4", "e7e5")),
    ("queen_pawn", ("d2d4", "d7d5")),
    ("english", ("c2c4", "e7e5")),
    ("reti", ("g1f3", "d7d5")),
    ("sicilian", ("e2e4", "c7c5")),
    ("indian", ("d2d4", "g8f6")),
)
RESULT_VALUES = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_positions(dataset_dir, split="test", max_positions=256, seed=1):
    """Uniform reservoir over playable target-player plies, preserving history.

    Every game is replayed, including opponents' moves. Rules this engine treats
    as terminal (including automatically accepted claimable draws) are excluded
    explicitly; a human PGN may legally continue after a claim became available.
    """
    if max_positions < 1:
        raise ValueError("max_positions must be positive")
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    if manifest.get("status") not in ("completed", "completed_with_rejections"):
        raise ValueError("Prepared dataset is not complete")
    if sha256_file(dataset_dir / "games.jsonl") != manifest.get("games_sha256"):
        raise ValueError("Prepared game records failed their SHA256 integrity check")
    rng = random.Random(seed)
    reservoir, candidate_count, game_count = [], 0, 0
    exclusions = Counter()
    with (dataset_dir / "games.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            game = json.loads(line)
            if game["split"] != split:
                continue
            game_count += 1
            if game["result"] not in RESULT_VALUES:
                raise ValueError(f"Unknown result in prepared game {game['game_id']}")
            state = GameState.from_fen(game["initial_fen"])
            for ply, uci in enumerate(game["moves"], 1):
                # Validate even moves not selected for evaluation.
                move = state.parse_uci(uci)
                if state.player == game["target_color"]:
                    terminal = terminal_details(state)
                    if terminal is None:
                        candidate_count += 1
                        slot = len(reservoir) if len(reservoir) < max_positions else rng.randrange(candidate_count)
                        if slot < max_positions:
                            row = {"game_id": game["game_id"], "game_index": game.get("game_index"),
                                   "ply": ply, "human_move": uci, "target_color": game["target_color"],
                                   "result": game["result"],
                                   "target_value": RESULT_VALUES[game["result"]] * (1 if state.player == "w" else -1),
                                   "state": state.fast_copy_for_search()}
                            if slot == len(reservoir):
                                reservoir.append(row)
                            else:
                                reservoir[slot] = row
                    else:
                        exclusions[terminal[0]] += 1
                state.make_move(move)
    if not reservoir:
        raise ValueError(f"No playable target-player positions in split {split!r}")
    reservoir.sort(key=lambda row: (str(row["game_id"]), row["ply"]))
    return reservoir, {"split": split, "games": game_count, "eligible_positions": candidate_count,
                       "sampled_positions": len(reservoir),
                       "sampled_games": len({row["game_id"] for row in reservoir}),
                       "engine_terminal_exclusions": dict(exclusions),
                       "method": "uniform seeded reservoir over target-player plies; whole-game history replayed"}


@torch.inference_mode()
def raw_decision(model, state):
    """One forward pass; probabilities and NLL are conditioned on legal moves."""
    started = time.perf_counter()
    legal = state.get_all_legal_moves(state.player, as_moves=True)
    if not legal:
        raise ValueError("raw_decision requires a playable position")
    device = next(model.parameters()).device
    logits, values = model(encode_gamestate(state).unsqueeze(0).to(device))
    actions = torch.tensor([move_to_action(move) for move in legal], device=device)
    log_probs = torch.log_softmax(logits[0, actions], dim=0).cpu().tolist()
    value = float(values.reshape(-1)[0].item())
    if not math.isfinite(value) or not all(math.isfinite(p) for p in log_probs):
        raise ValueError("Nonfinite network prediction")
    log_distribution = dict(zip(legal, log_probs))
    distribution = {move: math.exp(logp) for move, logp in log_distribution.items()}
    ordered = sorted(legal, key=lambda move: distribution[move], reverse=True)
    return {"move": ordered[0], "ordered": ordered, "distribution": distribution,
            "log_distribution": log_distribution, "value": value,
            "seconds": time.perf_counter() - started}


def search_decision(model, state, sims):
    started = time.perf_counter()
    distribution, root = mcts_search(state, model, num_sims=sims,
                                     add_root_dirichlet=False, return_root=True,
                                     root=None, batch_size=16)
    move = best_move(distribution, root)
    if move is None:
        raise ValueError("MCTS returned no move for an evaluation position")
    ordered = sorted(distribution, key=lambda candidate: (
        distribution[candidate], root.edges[candidate].q, root.edges[candidate].prior), reverse=True)
    return {"move": move, "ordered": ordered, "distribution": distribution,
            "seconds": time.perf_counter() - started}


def evaluate_position(model, example, sims=64):
    state = example["state"]
    raw, search = raw_decision(model, state), search_decision(model, state, sims)
    target = state.parse_uci(example["human_move"])
    raw_top = [move.uci() for move in raw["ordered"][:5]]
    search_top = [move.uci() for move in search["ordered"][:5]]
    return {
        **{key: value for key, value in example.items() if key != "state"},
        "fen": state.to_fen(), "phase": "opening" if example["ply"] <= 20 else "later",
        "legal_moves": len(raw["distribution"]),
        "raw_move": raw["move"].uci(), "search_move": search["move"].uci(),
        "raw_top5": raw_top, "search_top5": search_top,
        "raw_match": raw_top[0] == target.uci(), "search_match": search_top[0] == target.uci(),
        "raw_top5_match": target.uci() in raw_top, "search_top5_match": target.uci() in search_top,
        "move_changed": raw["move"] != search["move"],
        "human_move_probability": raw["distribution"][target],
        "human_move_nll": -raw["log_distribution"][target],
        "human_move_search_probability": search["distribution"].get(target, 0.0),
        "value": raw["value"], "value_squared_error": (raw["value"] - example["target_value"]) ** 2,
        "zero_value_squared_error": example["target_value"] ** 2,
        "raw_seconds": raw["seconds"], "search_seconds": search["seconds"],
        "search_simulations": sims,
    }


def position_summary(rows):
    def aggregate(selected):
        if not selected:
            return {"positions": 0}
        mean = lambda key: sum(row[key] for row in selected) / len(selected)
        return {"positions": len(selected), "raw_top1": mean("raw_match"),
                "search_top1": mean("search_match"), "raw_top5": mean("raw_top5_match"),
                "search_top5": mean("search_top5_match"), "move_change_fraction": mean("move_changed"),
                "raw_legal_nll": mean("human_move_nll"), "value_mse": mean("value_squared_error"),
                "zero_value_baseline_mse": mean("zero_value_squared_error"),
                "mean_raw_seconds": mean("raw_seconds"), "mean_search_seconds": mean("search_seconds")}
    return {"all": aggregate(rows), "opening": aggregate([r for r in rows if r["phase"] == "opening"]),
            "later": aggregate([r for r in rows if r["phase"] == "later"])}


def play_match(model, opening_name, opening_moves, mcts_color, sims=64, max_plies=160):
    if mcts_color not in ("w", "b") or max_plies < 1:
        raise ValueError("Invalid match color or ply budget")
    state = GameState()
    for uci in opening_moves:
        state.push_uci(uci)
    started, moves = time.perf_counter(), []
    raw_seconds = search_seconds = 0.0
    ending = terminal_details(state)
    for _ in range(max_plies):
        if ending is not None:
            break
        use_search = state.player == mcts_color
        choice = search_decision(model, state, sims) if use_search else raw_decision(model, state)
        if use_search:
            search_seconds += choice["seconds"]
        else:
            raw_seconds += choice["seconds"]
        moves.append(choice["move"].uci())
        state.make_move(choice["move"])
        ending = terminal_details(state)
    reason, result = ending if ending is not None else ("move_limit", None)
    outcome = "*" if result is None else "1-0" if result > 0 else "0-1" if result < 0 else "1/2-1/2"
    mcts_score = None if result is None else .5 if result == 0 else float((result > 0) == (mcts_color == "w"))
    return {"opening": opening_name, "opening_moves": list(opening_moves), "mcts_color": mcts_color,
            "policy_color": "b" if mcts_color == "w" else "w", "moves": moves,
            "plies_after_opening": len(moves), "total_plies": len(opening_moves) + len(moves),
            "max_plies_after_opening": max_plies, "outcome": outcome, "termination": reason,
            "truncated": ending is None, "mcts_score": mcts_score, "final_fen": state.to_fen(),
            "duration_seconds": time.perf_counter() - started,
            "raw_seconds": raw_seconds, "search_seconds": search_seconds,
            "search_simulations_per_move": sims}


def match_summary(matches):
    completed = [game for game in matches if game["mcts_score"] is not None]
    return {"games": len(matches), "completed_games": len(completed),
            "unresolved_cutoffs": len(matches) - len(completed),
            "mcts_wins": sum(game["mcts_score"] == 1 for game in completed),
            "policy_wins": sum(game["mcts_score"] == 0 for game in completed),
            "actual_draws": sum(game["mcts_score"] == .5 for game in completed),
            "mcts_score_on_completed_games": (sum(game["mcts_score"] for game in completed) / len(completed)
                                              if completed else None),
            "termination_counts": dict(Counter(game["termination"] for game in matches))}


def evaluate_checkpoint(checkpoint, dataset_dir, output_dir, *, split="test", max_positions=256,
                        seed=1, sims=64, threads=4, match_pairs=6, max_plies=160):
    if split not in ("validation", "test"):
        raise ValueError("Held-out evaluation requires validation or test split")
    if max_positions < 1 or sims < 1 or threads < 1 or not 0 <= match_pairs <= len(OPENINGS) or max_plies < 1:
        raise ValueError("Require positive sims/threads/max_plies and 0–6 match pairs")
    output_dir, dataset_dir, checkpoint = map(lambda path: Path(path).resolve(),
                                               (output_dir, dataset_dir, checkpoint))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
              "dataset_dir": str(dataset_dir), "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
              "split": split, "max_positions": max_positions, "seed": seed, "sims": sims,
              "threads": threads, "device": "cpu", "match_pairs": match_pairs, "max_plies": max_plies,
              "root_noise": False, "move_sampling": False, "search_tree_reuse": False,
              "started_at": utc_now(),
              "source_sha256": {path.name: sha256_file(path) for path in Path(__file__).parent.glob("*.py")}}
    write_json(output_dir / "config.json", config)
    started, rows, matches = time.perf_counter(), [], []
    summary = {"schema_version": 1, "status": "running", "config": config,
               "limitations": [
                   "Held-out human move agreement measures imitation, not move quality or strength.",
                   "Value labels are observed human outcomes, including resignation and clock effects.",
                   "The tactical suite and small deterministic paired matches do not estimate Elo.",
                   "Cutoff matches are unresolved and excluded from the completed-game score denominator.",
                   "MCTS uses more computation per move; this is not an equal-time comparison.",
                   "Search top-five ties are ordered by mean action value then prior; unvisited moves can appear at small budgets.",
               ]}
    write_json(output_dir / "summary.json", summary)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with torch.random.fork_rng(devices=[]):
            model = load_model(checkpoint, device="cpu")
            examples, sampling = sample_positions(dataset_dir, split, max_positions, seed)
            summary["sampling"] = sampling
            with (output_dir / "positions.jsonl").open("w") as stream:
                for index, example in enumerate(examples, 1):
                    record = evaluate_position(model, example, sims)
                    rows.append(record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    if index % 64 == 0 or index == len(examples):
                        print(f"Held-out positions: {index}/{len(examples)}", flush=True)
            probes = evaluate_probes(model, num_sims=sims)
            for probe in probes:
                expected = probe["expected_moves"]
                probe["raw_passed"] = probe["policy_best_move"] in expected if expected else None
                probe["search_passed"] = probe["passed"]
            summary["tactical_probes"] = {
                "scored_positions": sum(probe["raw_passed"] is not None for probe in probes),
                "raw_passes": sum(probe["raw_passed"] is True for probe in probes),
                "search_passes": sum(probe["search_passed"] is True for probe in probes), "positions": probes}
            with (output_dir / "matches.jsonl").open("w") as stream:
                for opening_name, opening_moves in OPENINGS[:match_pairs]:
                    for color in ("w", "b"):
                        game = play_match(model, opening_name, opening_moves, color, sims, max_plies)
                        matches.append(game)
                        stream.write(json.dumps(game, allow_nan=False) + "\n")
                        stream.flush()
                        print(f"Match {len(matches)}/{2 * match_pairs}: MCTS {color}; {game['termination']}; {game['outcome']}", flush=True)
            summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        torch.set_num_threads(previous_threads)
        summary.update(heldout=position_summary(rows), matches=match_summary(matches),
                       duration_seconds=time.perf_counter() - started, updated_at=utc_now())
        write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-positions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--match-pairs", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=160)
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(**vars(args))
    print(json.dumps({"status": result["status"], "heldout": result["heldout"]["all"],
                      "matches": result["matches"]}, indent=2))


if __name__ == "__main__":
    main()
