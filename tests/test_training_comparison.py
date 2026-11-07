"""Ablation comparisons use common value labels and retain provenance checks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from scr.chess_engine import GameState
    from scr import compare_training as comparison


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional training dependency")
class TrainingComparisonTests(unittest.TestCase):
    def test_common_outcome_labels_follow_side_to_move(self):
        labels = []
        for position in comparison.KNOWN_OUTCOME_POSITIONS:
            state = GameState.from_fen(position["fen"])
            expected = position["expected_value"]
            labels.append(expected)
            if expected == 1:
                self.assertFalse(state.is_checkmate())
                state.push_uci(position["mate_move"])
                self.assertTrue(state.is_checkmate())
                after = next(p for p in comparison.KNOWN_OUTCOME_POSITIONS
                             if p["id"] == position["id"] + "_after_mate")
                self.assertEqual(state.to_fen(), after["fen"])
            elif expected == -1:
                self.assertTrue(state.is_checkmate())
            else:
                self.assertEqual(state.draw_reason(), "insufficient_material")
        self.assertEqual(sorted(labels), [-1, -1, 0, 0, 1, 1])

    def test_config_comparison_ignores_only_intervention_and_output(self):
        baseline = {"seed": 1, "hidden": 8, "parameter_count": 123,
                    "run_dir": "old", "checkpoint_dir": "old/checkpoints", "progress": False}
        treatment = dict(baseline, cutoff_value_policy="exclude", run_dir="new",
                         checkpoint_dir="new/checkpoints", progress=True)
        self.assertEqual(comparison.config_differences(baseline, treatment), {})
        treatment.update(hidden=16, parameter_count=456, new_hyperparameter=2)
        self.assertEqual(set(comparison.config_differences(baseline, treatment)),
                         {"hidden", "parameter_count", "new_hyperparameter"})

    def test_sample_eligibility_legacy_fallback_and_masked_counts(self):
        legacy = {"batch_size": 8, "cutoff_target_fraction": .25}
        masked = dict(legacy, policy_sample_count=8, value_sample_count=6)
        counts = comparison.sample_counts([legacy, masked])
        self.assertEqual(counts, {"policy_samples": 16, "value_samples": 14,
                                 "cutoff_policy_samples": 4, "excluded_value_samples": 2,
                                 "cutoff_value_samples": 2})

    def make_runs(self, temporary):
        directories = [Path(temporary) / name for name in ("baseline", "treatment")]
        for index, directory in enumerate(directories):
            (directory / "checkpoints").mkdir(parents=True)
            config = {"iterations": 2, "seed": 1, "hidden": 8, "parameter_count": 1,
                      "run_dir": str(directory), "checkpoint_dir": str(directory / "checkpoints"),
                      "progress": bool(index)}
            if index:
                config["cutoff_value_policy"] = "exclude"
            (directory / "config.json").write_text(json.dumps({"config": config}))
            summary = {"status": "completed", "games_completed": 2, "optimizer_steps": 2,
                       "checkpoints": [{"path": f"/obsolete/machine/weights_{game}.pt", "game": game}
                                       for game in (0, 2)]}
            (directory / "summary.json").write_text(json.dumps(summary))
            games = [
                {"game": 1, "plies": 2, "moves": ["f2f3", "e7e5"], "termination": "checkmate",
                 "truncated": False, "outcome": "0-1"},
                {"game": 2, "plies": 2, "moves": ["e2e4", "e7e5"], "termination": "move_limit",
                 "truncated": True, "outcome": "*"},
            ]
            updates = [{"game": 2, "step": step, "batch_size": 4, "cutoff_target_fraction": .5,
                        "policy_loss": 6.0 - step, "policy_kl": 3.0 - step} for step in (1, 2)]
            if index:
                for update in updates:
                    update.update(policy_sample_count=4, value_sample_count=2)
            probe = {"id": "mate", "value": .1, "passed": True,
                     "policy_probability_expected": .2, "search_probability_expected": .8}
            probes = [{"game": game, "step": game, "duration_seconds": 1 + index,
                       "positions": [probe]} for game in (0, 2)]
            for name, rows in (("games", games), ("updates", updates), ("probes", probes)):
                (directory / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            for game, value in ((0, 0.), (2, .2 if index == 0 else -.3)):
                torch.save({"state_dict": {"constant": torch.tensor(value)}},
                           directory / "checkpoints" / f"weights_{game}.pt")
        return directories

    @staticmethod
    def fake_load_model(path, device):
        if device != "cpu":
            raise AssertionError("comparison must use CPU")
        torch.rand(1)  # Loading a real module also consumes RNG.
        value = comparison._state_dict(path)["constant"]

        class ConstantModel(torch.nn.Module):
            def forward(self, states):
                if states.shape != (6, 20, 8, 8):
                    raise AssertionError("all six common positions must be evaluated")
                return torch.zeros(len(states), 1), value.expand(len(states), 1)

        return ConstantModel()

    def test_comparison_counts_provenance_common_mse_and_portable_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(comparison, "load_model", self.fake_load_model):
            baseline, treatment = self.make_runs(temporary)
            output = Path(temporary) / "comparison.json"
            before = torch.get_rng_state().clone()
            result = comparison.compare_runs(baseline, treatment, output)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(result["complete"])
            self.assertTrue(result["valid_comparison"], result)
            self.assertEqual(result, json.loads(output.read_text()))
            self.assertEqual(result["pre_update_games_compared"], 2)
            self.assertEqual(result["baseline"]["value_samples"], 8)
            self.assertEqual(result["baseline"]["cutoff_value_samples"], 4)
            self.assertEqual(result["treatment"]["value_samples"], 4)
            self.assertEqual(result["treatment"]["cutoff_value_samples"], 0)
            self.assertEqual(result["treatment"]["excluded_value_samples"], 4)
            self.assertEqual(result["treatment"]["policy_samples"], 8)
            self.assertEqual(result["treatment"]["actually_ended_games"], 1)
            self.assertEqual(result["treatment"]["cutoff_positions"], 2)
            self.assertEqual(result["treatment"]["last_probe"]["tactical_passes"], 1)
            self.assertEqual(result["baseline"]["policy_kl_last"], 1.)
            diagnostic = result["common_value_diagnostics"]
            self.assertTrue(diagnostic["evaluation_only"])
            self.assertAlmostEqual(diagnostic["zero_baseline_mse"], 2 / 3)
            for name, offset in (("initial", 0.), ("baseline", .2), ("treatment", -.3)):
                self.assertAlmostEqual(diagnostic["mse"][name], 2 / 3 + offset ** 2, places=6)
            self.assertTrue(all("expected_value" in row and "fen" in row for row in diagnostic["positions"]))
            with self.assertRaisesRegex(ValueError, "outside both raw run directories"):
                comparison.compare_runs(baseline, treatment, baseline / "summary.json")

    def test_incomplete_or_mismatched_run_is_recorded_as_invalid(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(comparison, "load_model", self.fake_load_model):
            baseline, treatment = self.make_runs(temporary)
            metadata = json.loads((treatment / "config.json").read_text())
            metadata["config"]["seed"] = 2
            (treatment / "config.json").write_text(json.dumps(metadata))
            summary = json.loads((treatment / "summary.json").read_text())
            summary["status"] = "interrupted"
            (treatment / "summary.json").write_text(json.dumps(summary))
            torch.save({"constant": torch.tensor(.01)}, treatment / "checkpoints/weights_0.pt")
            result = comparison.compare_runs(baseline, treatment)
            self.assertFalse(result["valid_comparison"])
            self.assertFalse(result["complete"])
            self.assertFalse(result["checks"]["hyperparameters_equal"])
            self.assertFalse(result["checks"]["initial_weights_exact"])
            self.assertEqual(result["hyperparameter_differences"], {"seed": {"baseline": 1, "treatment": 2}})

    def test_missing_checkpoint_is_explicit_and_blocks_valid_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline, treatment = self.make_runs(temporary)
            (treatment / "checkpoints/weights_2.pt").unlink()
            result = comparison.compare_runs(baseline, treatment)
            self.assertFalse(result["valid_comparison"])
            self.assertFalse(result["checks"]["checkpoints_available"])
            self.assertIsNone(result["common_value_diagnostics"])
            self.assertIn("Checkpoint missing", result["errors"][0])

    def test_divergence_before_training_and_initial_probe_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(comparison, "load_model", self.fake_load_model):
            baseline, treatment = self.make_runs(temporary)
            path = treatment / "games.jsonl"
            path.write_text(path.read_text().replace("f2f3", "g2g3"))
            path = treatment / "probes.jsonl"
            path.write_text(path.read_text().replace('"value": 0.1', '"value": 0.2'))
            path = treatment / "updates.jsonl"
            path.write_text(path.read_text().replace('"policy_loss": 5.0', '"policy_loss": 5.1'))
            result = comparison.compare_runs(baseline, treatment)
            self.assertFalse(result["valid_comparison"])
            self.assertFalse(result["checks"]["games_before_first_update_moves_equal"])
            self.assertFalse(result["checks"]["initial_probe_equal_excluding_timing"])
            self.assertFalse(result["checks"]["first_policy_loss_close"])


if __name__ == "__main__":
    unittest.main()
