"""Run logs distinguish actual outcomes, training labels, and partial runs."""
import importlib.util
import hashlib
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
    def test_checkpoint_initialization_preserves_weights_and_starts_fresh_rl_stage(self):
        import torch
        from scr.chess_engine import GameState
        from scr.main import main
        from scr.rl_interface import PolicyValueNet, encode_gamestate, load_model, move_to_action, NUM_ACTIONS
        from scr.train import train_step

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, directory = root / "supervised.pt", root / "rl"
            torch.set_num_threads(1)
            torch.manual_seed(7)
            model = PolicyValueNet(hidden=4)
            # A supervised checkpoint may contain optimizer/RNG/training state;
            # this entry point deliberately imports only its network weights.
            torch.save({"format_version": 2, "state_dict": model.state_dict(),
                        "optimizer_state_dict": {"not_for_this_stage": True},
                        "training_state": {"completed_epoch": 3, "step": 999}}, checkpoint)
            main(["--resume", str(checkpoint), "--iterations", "1", "--sims", "1", "--max-moves", "1",
                  "--batch-size", "1", "--min-buffer", "1", "--updates", "1", "--threads", "1",
                  "--device", "cpu", "--seed", "2", "--learning-rate", "0.0001",
                  "--run-dir", str(directory), "--probe-every", "0", "--trace-every", "0"])
            metadata = json.loads((directory / "config.json").read_text())
            config = metadata["config"]
            self.assertEqual(config["initialization"], "checkpoint_weights")
            self.assertEqual(config["initialization_checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(config["optimizer_initialization"], "fresh Adam")
            self.assertEqual(config["replay_initialization"], "empty")
            self.assertEqual(config["hidden"], 4)
            self.assertIn("both players", config["policy_target"])
            self.assertIn("side-to-move", config["value_target"])
            initial = load_model(directory / "checkpoints/policy_value_net_0.pt", "cpu")
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, initial.state_dict()[name]), name)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["optimizer_steps"], 1)
            self.assertEqual(summary["replay"]["positions"], 1)
            game = json.loads((directory / "games.jsonl").read_text())
            self.assertEqual(game["step_before_updates"], 0)
            self.assertEqual(game["termination"], "move_limit")
            # One simulation assigns all visits to the single played move.
            state = GameState()
            policy = torch.zeros(NUM_ACTIONS)
            policy[move_to_action(state.parse_uci(game["moves"][0]))] = 1.
            train_step(initial, torch.optim.Adam(initial.parameters(), lr=.0001),
                       [(encode_gamestate(state), policy, 0.)], "cpu")
            final = load_model(directory / "checkpoints/policy_value_net_1.pt", "cpu")
            for name, tensor in initial.state_dict().items():
                self.assertTrue(torch.equal(tensor, final.state_dict()[name]), name)

    def test_training_runner_defaults_to_draw_labels_and_writes_inspectable_outputs(self):
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
            self.assertEqual(summary["replay"]["policy_target_positions"], 4)
            self.assertEqual(summary["replay"]["value_target_positions"], 4)
            self.assertEqual(summary["replay"]["excluded_value_positions"], 0)
            self.assertEqual(summary["policy_samples_seen"], 4)
            self.assertEqual(summary["value_samples_seen"], 4)
            self.assertEqual(summary["cutoff_value_policy"], "draw")
            games = [json.loads(line) for line in (directory / "games.jsonl").read_text().splitlines()]
            self.assertEqual(len(games[0]["trace"]), 2)
            self.assertTrue(all(row["value_target_available"] is True for row in games))
            updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["cutoff_target_fraction"] == 1 for row in updates))
            self.assertTrue(all(row["policy_sample_count"] == 2 for row in updates))
            self.assertTrue(all(row["value_sample_count"] == 2 for row in updates))
            self.assertTrue(all(row["value_metrics_available"] for row in updates))
            self.assertTrue(all(row["value_mae"] is not None for row in updates))
            self.assertTrue(all(row["target_value_mean"] == 0 for row in updates))
            self.assertTrue(all(row["zero_value_baseline_mse"] == 0 for row in updates))
            self.assertTrue(all(row["value_target_draw_fraction"] == 1 for row in updates))
            self.assertTrue((directory / "report.html").is_file())
            self.assertTrue((directory / "checkpoints/policy_value_net_0.pt").is_file())
            self.assertTrue((directory / "checkpoints/policy_value_net_2.pt").is_file())

    def test_explicit_experimental_exclusion_keeps_cutoffs_for_policy_only(self):
        from scr.main import main
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "value-mask"
            main(["--iterations", "1", "--sims", "2", "--max-moves", "2",
                  "--hidden", "8", "--batch-size", "2", "--min-buffer", "1",
                  "--updates", "1", "--threads", "1", "--device", "cpu",
                  "--run-dir", str(directory), "--probe-every", "0", "--trace-every", "0",
                  "--cutoff-value-policy", "exclude"])
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["cutoff_value_policy"], "exclude")
            self.assertEqual(summary["excluded_value_positions"], 2)
            self.assertEqual(summary["replay"]["policy_target_positions"], 2)
            self.assertEqual(summary["replay"]["value_target_positions"], 0)
            self.assertEqual(summary["policy_samples_seen"], 2)
            self.assertEqual(summary["value_samples_seen"], 0)
            game = json.loads((directory / "games.jsonl").read_text())
            self.assertFalse(game["value_target_available"])
            update = json.loads((directory / "updates.jsonl").read_text())
            self.assertEqual(update["policy_sample_count"], 2)
            self.assertEqual(update["value_sample_count"], 0)
            self.assertEqual(update["value_loss"], 0)
            self.assertFalse(update["value_metrics_available"])
            self.assertIsNone(update["value_mae"])
            self.assertIsNone(update["zero_value_baseline_mse"])
            self.assertIsNone(update["value_target_draw_fraction"])


if __name__ == "__main__":
    unittest.main()
