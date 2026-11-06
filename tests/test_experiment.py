"""Run logs distinguish actual outcomes, training labels, and partial runs."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chess"))
from scr.experiment import RunLogger


class RunLoggerTests(unittest.TestCase):
    def test_summary_counts_actual_draws_separately_from_cutoff_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = RunLogger(Path(temporary) / "run", {"seed": 7}, torch_version="test", source_root=ROOT)
            for index, (termination, outcome, plies, result) in enumerate([
                ("move_limit", "*", 200, 0), ("stalemate", "1/2-1/2", 40, 0),
                ("checkmate", "1-0", 25, 1),
            ], 1):
                logger.game({"game": index, "termination": termination, "outcome": outcome,
                             "truncated": termination == "move_limit", "plies": plies,
                             "result": result, "duration_seconds": 1.0, "trace": []})
            logger.update({"step": 1, "loss": 2.0})
            summary = logger.refresh("completed")
            self.assertEqual(summary["actual_draws"], 1)
            self.assertEqual(summary["decisive_games"], 1)
            self.assertEqual(summary["move_limit_games"], 1)
            self.assertAlmostEqual(summary["move_limit_fraction"], 1 / 3)
            self.assertAlmostEqual(summary["cutoff_target_fraction"], 200 / 265)
            self.assertEqual(summary["optimizer_steps"], 1)
            self.assertEqual(len((logger.directory / "games.jsonl").read_text().splitlines()), 3)
            metadata = json.loads((logger.directory / "config.json").read_text())
            self.assertTrue(metadata["source_sha256"])
            self.assertTrue((logger.directory / "source/chess/scr/main.py").is_file())

    def test_failed_run_retains_records_and_nonempty_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            logger = RunLogger(directory, {}, torch_version="test", source_root=ROOT)
            logger.update({"step": 1, "loss": 1.0})
            logger.refresh("failed", error="test failure")
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["optimizer_steps"], 1)
            with self.assertRaises(ValueError):
                RunLogger(directory, {}, torch_version="test", source_root=ROOT)
            self.assertEqual(json.loads((directory / "updates.jsonl").read_text())["step"], 1)
            with self.assertRaises(ValueError):
                logger.update({"loss": float("nan")})


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch is not installed")
class TrainingRunTests(unittest.TestCase):
    def test_training_runner_writes_inspectable_outputs(self):
        from scr.main import main
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            main(["--iterations", "2", "--sims", "2", "--max-moves", "2",
                  "--hidden", "8", "--batch-size", "2", "--min-buffer", "1",
                  "--updates", "1", "--threads", "1", "--device", "cpu",
                  "--run-dir", str(directory), "--probe-every", "0", "--trace-every", "1"])
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["termination_counts"], {"move_limit": 2})
            self.assertEqual(summary["optimizer_steps"], 2)
            self.assertEqual(summary["replay"]["cutoff_positions"], 4)
            games = [json.loads(line) for line in (directory / "games.jsonl").read_text().splitlines()]
            self.assertEqual(len(games[0]["trace"]), 2)
            updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["cutoff_target_fraction"] == 1 for row in updates))
            self.assertTrue((directory / "report.html").is_file())
            self.assertTrue((directory / "checkpoints/policy_value_net_0.pt").is_file())
            self.assertTrue((directory / "checkpoints/policy_value_net_2.pt").is_file())


if __name__ == "__main__":
    unittest.main()
