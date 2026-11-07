"""Compare a legacy draw-label run with a cutoff-value-exclusion run.

The six common value examples are evaluation diagnostics only. They are never
added to replay or used for optimization. Run with ``python -m scr.compare_training``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch

try:
    from .chess_engine import GameState
    from .rl_interface import encode_gamestate, load_model
except ImportError:
    from chess_engine import GameState
    from rl_interface import encode_gamestate, load_model


KNOWN_OUTCOME_POSITIONS = (
    {"id": "white_mate_in_one", "fen": "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
     "expected_value": 1.0, "mate_move": "g6g7"},
    {"id": "white_mate_in_one_after_mate", "fen": "7k/6Q1/5K2/8/8/8/8/8 b - - 1 1",
     "expected_value": -1.0},
    {"id": "black_mate_in_one", "fen": "8/8/8/8/8/5kq1/8/7K b - - 0 1",
     "expected_value": 1.0, "mate_move": "g3g2"},
    {"id": "black_mate_in_one_after_mate", "fen": "8/8/8/8/8/5k2/6q1/7K w - - 1 2",
     "expected_value": -1.0},
    {"id": "bare_kings_w", "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1", "expected_value": 0.0},
    {"id": "bare_kings_b", "fen": "4k3/8/8/8/8/8/8/4K3 b - - 0 1", "expected_value": 0.0},
)
IGNORED_CONFIG_KEYS = {"cutoff_value_policy", "run_dir", "checkpoint_dir", "progress"}


def config_differences(baseline, treatment):
    """Compare every recorded hyperparameter except the intervention and output UI."""
    return {key: {"baseline": baseline.get(key), "treatment": treatment.get(key)}
            for key in sorted(baseline.keys() | treatment.keys())
            if key not in IGNORED_CONFIG_KEYS and baseline.get(key) != treatment.get(key)}


def sample_counts(updates):
    """Count sampled rows, including repeats; old logs trained value on every row."""
    policy = value = cutoff_policy = 0
    for update in updates:
        batch = int(update.get("policy_sample_count", update["batch_size"]))
        policy += batch
        value += int(update.get("value_sample_count", batch))
        cutoff_policy += round(float(update.get("cutoff_target_fraction", 0)) * batch)
    excluded = policy - value
    return {"policy_samples": policy, "value_samples": value,
            "cutoff_policy_samples": cutoff_policy, "excluded_value_samples": excluded,
            "cutoff_value_samples": cutoff_policy - excluded}


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_run(directory):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    return {"directory": directory, "metadata": metadata, "config": metadata["config"],
            "summary": summary, **{name: _jsonl(directory / f"{name}.jsonl")
                                    for name in ("games", "updates", "probes")}}


def _run_stats(run):
    games, updates, summary, config = (run[key] for key in ("games", "updates", "summary", "config"))
    cutoff = [game for game in games if game.get("truncated", game.get("termination") == "move_limit")]
    complete = (summary.get("status") == "completed"
                and len(games) == summary.get("games_completed") == config.get("iterations")
                and len(updates) == summary.get("optimizer_steps"))
    probes = run["probes"][-1] if run["probes"] else None
    tactical = [position for position in probes["positions"] if position.get("passed") is not None] if probes else []
    return {
        "directory": str(run["directory"]), "status": summary.get("status"), "complete": complete,
        "cutoff_value_policy": config.get("cutoff_value_policy", "draw"),
        "games": len(games), "actually_ended_games": len(games) - len(cutoff), "cutoff_games": len(cutoff),
        "actual_draws": sum(game.get("outcome") == "1/2-1/2" for game in games),
        "termination_counts": dict(Counter(game["termination"] for game in games)),
        "positions": sum(game["plies"] for game in games),
        "cutoff_positions": sum(game["plies"] for game in cutoff), "optimizer_steps": len(updates),
        **sample_counts(updates),
        "policy_kl_first": updates[0].get("policy_kl") if updates else None,
        "policy_kl_last": updates[-1].get("policy_kl") if updates else None,
        "policy_loss_first": updates[0].get("policy_loss") if updates else None,
        "policy_loss_last": updates[-1].get("policy_loss") if updates else None,
        "last_probe": {"game": probes["game"], "tactical_passes": sum(bool(p["passed"]) for p in tactical),
                       "tactical_count": len(tactical),
                       "positions": [{key: p.get(key) for key in
                                      ("id", "value", "passed", "policy_probability_expected",
                                       "search_probability_expected")} for p in probes["positions"]]} if probes else None,
    }


def _without_timing(value):
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items()
                if not key.endswith("_seconds")}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _initial_probe(run):
    return next((_without_timing(probe) for probe in run["probes"] if probe.get("game") == 0), None)


def _checkpoint(run, game):
    entry = next((entry for entry in reversed(run["summary"].get("checkpoints", []))
                  if entry["game"] == game), None)
    if entry is None:
        raise FileNotFoundError(f"No checkpoint recorded at game {game} in {run['directory']}")
    saved = Path(entry["path"])
    candidates = [saved] if saved.is_absolute() else [run["directory"] / saved]
    candidates.append(run["directory"] / "checkpoints" / saved.name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Checkpoint missing: {saved.name} in {run['directory']}")


def _state_dict(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)


def _states_equal(left, right):
    return (left.keys() == right.keys()
            and all(left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
                    and torch.equal(left[key], right[key]) for key in left))


def evaluate_known_outcomes(checkpoints):
    """Raw network values on identical, labeled positions; no search or training."""
    states = torch.stack([encode_gamestate(GameState.from_fen(p["fen"])) for p in KNOWN_OUTCOME_POSITIONS])
    predictions = {}
    # Loading a model initializes its module before replacing weights; keep the
    # caller's RNG untouched so diagnostics cannot change a later training run.
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        for name, path in checkpoints.items():
            model = load_model(path, device="cpu")
            predictions[name] = model(states)[1].flatten().tolist()
    positions = [{"id": position["id"], "fen": position["fen"],
                  "expected_value": position["expected_value"],
                  **{name: values[index] for name, values in predictions.items()}}
                 for index, position in enumerate(KNOWN_OUTCOME_POSITIONS)]
    return {"value_perspective": "player to move", "evaluation_only": True,
            "positions": positions,
            "mse": {name: sum((p[name] - p["expected_value"]) ** 2 for p in positions) / len(positions)
                    for name in predictions},
            "zero_baseline_mse": sum(p["expected_value"] ** 2 for p in positions) / len(positions)}


def compare_runs(baseline_dir, treatment_dir, output=None):
    """Return comparison evidence, preserving incomplete/mismatched runs as invalid.

    Missing or malformed config/summary JSON raises an error. Missing checkpoints
    and incomplete logs are reported in the result instead of silently passing.
    Only an explicitly supplied output path is written; raw runs are never edited.
    """
    baseline, treatment = _read_run(baseline_dir), _read_run(treatment_dir)
    baseline_stats, treatment_stats = _run_stats(baseline), _run_stats(treatment)
    differences = config_differences(baseline["config"], treatment["config"])
    initial_probes = [_initial_probe(run) for run in (baseline, treatment)]
    first_update_games = [run["updates"][0]["game"] if run["updates"] else None
                          for run in (baseline, treatment)]
    prefix_count = first_update_games[0] if first_update_games[0] == first_update_games[1] else None
    warmup_moves_equal = (prefix_count is not None and prefix_count > 0
                         and len(baseline["games"]) >= prefix_count and len(treatment["games"]) >= prefix_count
                         and all(a["moves"] == b["moves"] for a, b in
                                 zip(baseline["games"][:prefix_count], treatment["games"][:prefix_count])))
    first_losses = [stats["policy_loss_first"] for stats in (baseline_stats, treatment_stats)]
    checks = {
        "runs_complete": baseline_stats["complete"] and treatment_stats["complete"],
        "hyperparameters_equal": not differences,
        "cutoff_policy_intervention": (baseline_stats["cutoff_value_policy"] == "draw"
                                       and treatment_stats["cutoff_value_policy"] == "exclude"),
        "initial_weights_exact": False,
        "initial_probe_equal_excluding_timing": initial_probes[0] is not None and initial_probes[0] == initial_probes[1],
        "games_before_first_update_moves_equal": warmup_moves_equal,
        "first_policy_loss_close": all(value is not None for value in first_losses)
                                   and math.isclose(*first_losses, rel_tol=1e-6, abs_tol=1e-7),
        "checkpoints_available": False,
    }
    errors, checkpoints, common_values = [], {}, None
    try:
        for name, run in (("baseline", baseline), ("treatment", treatment)):
            checkpoints[name + "_initial"] = _checkpoint(run, 0)
            checkpoints[name + "_final"] = _checkpoint(run, run["config"]["iterations"])
        checks["checkpoints_available"] = True
        checks["initial_weights_exact"] = _states_equal(_state_dict(checkpoints["baseline_initial"]),
                                                        _state_dict(checkpoints["treatment_initial"]))
        common_values = evaluate_known_outcomes({"initial": checkpoints["baseline_initial"],
                                                "baseline": checkpoints["baseline_final"],
                                                "treatment": checkpoints["treatment_final"]})
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result = {
        "schema_version": 1, "complete": checks["runs_complete"],
        "valid_comparison": all(checks.values()) and not errors, "checks": checks,
        "hyperparameter_differences": differences,
        "pre_update_games_compared": prefix_count,
        "baseline": baseline_stats, "treatment": treatment_stats,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "common_value_diagnostics": common_values, "errors": errors,
        "limitations": [
            "One small seeded run per condition cannot establish a general strength or Elo improvement.",
            "After the first changed update, self-play trajectories and replay populations can diverge.",
            "Logged value-loss averages use different target populations and must not be compared as an accuracy test.",
            "The six shared synthetic value positions are a small diagnostic, include terminal states, and are never trained on.",
            "Policy KL is measured on each run's own sampled search targets, not a shared held-out policy dataset.",
            "Sample counts include repeated replay draws; they are not counts of unique positions.",
        ],
    }
    if output is not None:
        output = Path(output)
        # Do not let an output option overwrite either run's source artifacts.
        if any(output.resolve().is_relative_to(run["directory"]) for run in (baseline, treatment)):
            raise ValueError("Comparison output must be outside both raw run directories")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("treatment_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Write comparison JSON outside the raw runs")
    args = parser.parse_args(argv)
    result = compare_runs(args.baseline_dir, args.treatment_dir, args.output)
    if args.output:
        print(json.dumps({"output": str(args.output), "complete": result["complete"],
                          "valid_comparison": result["valid_comparison"], "errors": result["errors"]}))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
