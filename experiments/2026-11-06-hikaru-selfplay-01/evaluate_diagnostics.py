"""Measure fixed checkpoint curves and human-task retention after self-play.

Run from the repository root with .venv/bin/python. These are retrospective
diagnostics; the fixed game-100 candidate is never selected by these scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "chess"))

import torch

from scr.evaluate_models import compare_values
from scr.experiment import write_json
from scr.human_data import iter_batches
from scr.rl_interface import load_model
from scr.supervised import evaluate_batches


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/2026-11-06-hikaru-selfplay-01")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "runs/2026-11-06-hikaru-value-benchmark-01")
    parser.add_argument("--human-dataset-dir", type=Path, default=ROOT / "data/hikaru-blitz-5000/prepared")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    run = args.run_dir.resolve()
    config = json.loads((run / "config.json").read_text())["config"]
    summary = json.loads((run / "summary.json").read_text())
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    if summary["status"] != "completed" or summary["games_completed"] != protocol["training"]["iterations"]:
        raise ValueError("Diagnostics require the completed fixed-budget training run")
    baseline_path = Path(config["resume"])
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path
    baseline_hash = sha256(baseline_path)
    if baseline_hash != protocol["baseline_sha256"] or baseline_hash != config["initialization_checkpoint_sha256"]:
        raise ValueError("The supervised baseline differs from the recorded protocol")
    checkpoints = {game: run / f"checkpoints/policy_value_net_{game}.pt"
                   for game in protocol["value_curve_games"]}
    input_hashes = {str(path.resolve()): sha256(path) for path in (
        baseline_path, protocol_path, run / "config.json", run / "summary.json",
        args.benchmark_dir / "manifest.json", args.benchmark_dir / "games.jsonl",
        args.human_dataset_dir / "manifest.json", *checkpoints.values())}
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "selection": protocol["candidate_selection"], "device": "cpu", "threads": args.threads,
        "batch_size": 128, "bootstrap_samples": 2000, "bootstrap_seed": 1,
        "input_sha256": input_hashes, "script_sha256": sha256(__file__),
        "torch_version": str(torch.__version__), "human_split": "test",
        "population_note": "Frozen self-play benchmark and human holdout remain separate; neither selects the candidate.",
    }
    write_json(run / "diagnostics_config.json", metadata)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1)
    baseline = load_model(baseline_path, device="cpu")
    curve = []
    for game, path in checkpoints.items():
        candidate = load_model(path, device="cpu")
        if game == 0 and not all(torch.equal(tensor, candidate.state_dict()[name])
                                 for name, tensor in baseline.state_dict().items()):
            raise ValueError("The starting self-play weights do not equal the supervised model")
        metrics = compare_values(baseline, candidate, args.benchmark_dir, batch_size=128,
                                 bootstrap_samples=2000, seed=1)
        curve.append({"game": game, "checkpoint": str(path), "checkpoint_sha256": sha256(path),
                      "metrics": metrics})
        write_json(run / "value_curve.json", curve)
        print(f"Game {game}: fixed-benchmark value MSE {metrics['all']['candidate']['mse']}", flush=True)
    final_game = protocol["training"]["iterations"]
    if protocol["value_curve_games"][-1] != final_game:
        raise ValueError("The last diagnostic checkpoint must be the fixed final candidate")
    retention = {"config": {"dataset": str(args.human_dataset_dir.resolve()), "split": "test",
                             "dataset_manifest_sha256": sha256(args.human_dataset_dir / "manifest.json"),
                             "policy_population": "Hikaru turns only; full-action probabilities",
                             "value_population": "both players; recorded human outcomes",
                             "used_for_selection": False},
                 "baseline": {"checkpoint": str(baseline_path), "checkpoint_sha256": baseline_hash},
                 "candidate": {"checkpoint": str(checkpoints[final_game]),
                               "checkpoint_sha256": sha256(checkpoints[final_game])}}
    for name, model in (("baseline", baseline), ("candidate", candidate)):
        retention[name]["metrics"] = evaluate_batches(
            model, iter_batches(args.human_dataset_dir, "test", 128), device="cpu")
        print(f"Human test {name}: {retention[name]['metrics']}", flush=True)
    write_json(run / "human_retention.json", retention)
    if any(sha256(path) != value for path, value in input_hashes.items()):
        raise ValueError("A diagnostic input changed during evaluation")
    metadata.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
    write_json(run / "diagnostics_config.json", metadata)


if __name__ == "__main__":
    main()
