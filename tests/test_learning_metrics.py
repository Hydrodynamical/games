"""Metrics must describe learning faithfully without changing its update."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from scr.chess_engine import GameState
    from scr.diagnostics import PROBES, evaluate_probes
    from scr.rl_interface import NUM_ACTIONS, PolicyValueNet
    from scr.train import train_step


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional training dependency")
class LearningMetricTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make_model(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([.4, -.2, .1]))
                self.value = torch.nn.Parameter(torch.tensor(.25))

            def forward(self, states):
                return self.logits.expand(len(states), -1), self.value.expand(len(states), 1)

        return TinyModel()

    def test_metrics_match_preupdate_predictions_and_standard_optimizer(self):
        model = self.make_model()
        reference = deepcopy(model)
        batch = [(torch.zeros(1), torch.tensor([.75, .25, 0.]), 1.),
                 (torch.ones(1), torch.tensor([0., 0., 1.]), -1.)]
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
        targets = torch.stack([item[1] for item in batch])
        values = torch.tensor([1., -1.])
        logits, predicted = reference(torch.stack([item[0] for item in batch]))
        expected_policy = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        expected_value = (predicted.squeeze(1) - values).square().mean()
        reference_optimizer.zero_grad()
        (expected_policy + expected_value).backward()
        expected_gradient = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in reference.parameters()))
        reference_optimizer.step()
        metrics = train_step(model, optimizer, batch, "cpu")
        entropy = -(.75 * math.log(.75) + .25 * math.log(.25)) / 2
        self.assertAlmostEqual(metrics["policy_loss"], float(expected_policy.detach()), places=6)
        self.assertAlmostEqual(metrics["policy_target_entropy"], entropy, places=6)
        self.assertAlmostEqual(metrics["policy_kl"], metrics["policy_loss"] - entropy, places=6)
        self.assertAlmostEqual(metrics["policy_top1_agreement"], .5)
        self.assertAlmostEqual(metrics["predicted_value_mean"], .25)
        self.assertAlmostEqual(metrics["predicted_value_std"], 0)
        self.assertAlmostEqual(metrics["target_value_mean"], 0)
        self.assertAlmostEqual(metrics["target_value_std"], 1)
        self.assertAlmostEqual(metrics["value_mae"], 1)
        self.assertAlmostEqual(metrics["value_loss"], 1.0625)
        self.assertAlmostEqual(metrics["zero_value_baseline_mse"], 1)
        self.assertAlmostEqual(metrics["gradient_norm"], expected_gradient, places=6)
        self.assertAlmostEqual(metrics["learning_rate"], .03)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(all(isinstance(value, float) and math.isfinite(value) for value in metrics.values()))
        json.dumps(metrics, allow_nan=False)

    def test_single_example_std_is_zero(self):
        model = self.make_model()
        metrics = train_step(model, torch.optim.SGD(model.parameters(), lr=.01),
                             [(torch.zeros(1), torch.tensor([1., 0., 0.]), 0.)], "cpu")
        self.assertEqual(metrics["predicted_value_std"], 0)
        self.assertEqual(metrics["target_value_std"], 0)
        self.assertEqual(metrics["zero_value_baseline_mse"], 0)
        json.dumps(metrics, allow_nan=False)

    def test_zero_value_baseline_uses_squared_targets(self):
        model = self.make_model()
        batch = [(torch.zeros(1), torch.tensor([1., 0., 0.]), value)
                 for value in (1., 0., -.5)]
        metrics = train_step(model, torch.optim.SGD(model.parameters(), lr=.01), batch, "cpu")
        self.assertAlmostEqual(metrics["zero_value_baseline_mse"], (1 + 0 + .25) / 3, places=6)

    def test_nonfinite_predictions_do_not_update_parameters(self):
        model = self.make_model()
        with torch.no_grad():
            model.value.fill_(float("nan"))
        before = model.logits.detach().clone()
        with self.assertRaisesRegex(ValueError, "predictions must be finite"):
            train_step(model, torch.optim.SGD(model.parameters(), lr=.01),
                       [(torch.zeros(1), torch.tensor([1., 0., 0.]), 0.)], "cpu")
        self.assertTrue(torch.equal(model.logits, before))


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional training dependency")
class FixedProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_tactical_references_are_legal_and_baseline_solves_them(self):
        records = evaluate_probes(None, num_sims=80)
        self.assertEqual(len(records), len(PROBES))
        self.assertEqual(len({record["id"] for record in records}), len(records))
        json.dumps(records, allow_nan=False)
        for record in records:
            with self.subTest(probe=record["id"]):
                position = GameState.from_fen(record["fen"])
                for uci in record["expected_moves"]:
                    move = position.parse_uci(uci)
                    if "mate" in record["id"]:
                        position.make_move(move)
                        self.assertTrue(position.is_checkmate())
                        position.undo_last_move()
                    elif "capture" in record["id"]:
                        self.assertTrue(move.piece_captured.endswith("Q"))
                if record["expected_moves"]:
                    self.assertTrue(record["passed"])
                else:
                    self.assertIsNone(record["passed"])
        bare_kings = next(record for record in records if record["id"] == "bare_kings")
        self.assertIsNone(bare_kings["search_best_move"])
        self.assertEqual(bare_kings["expected_value"], 0)

    def test_neural_probes_preserve_weights_modes_and_rng(self):
        class StochasticProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.child = torch.nn.Dropout(.5)

            def forward(self, states):
                # Deliberately consume RNG even in eval to test isolation.
                random.random()
                torch.rand(1)
                logits = torch.zeros(len(states), NUM_ACTIONS, device=states.device) + self.anchor
                value = torch.full((len(states), 1), .25, device=states.device) + self.anchor
                return logits, value

        model = StochasticProbeModel()
        model.train()
        model.child.eval()
        before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
        first = evaluate_probes(model, num_sims=3)
        second = evaluate_probes(model, num_sims=3)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        for parameter, original in zip(model.parameters(), before_parameters):
            self.assertTrue(torch.equal(parameter, original))
        self.assertTrue(all(record["value"] == .25 for record in first))
        json.dumps(first, allow_nan=False)

    def test_model_failure_restores_evaluation_state(self):
        class BrokenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, states):
                random.random()
                torch.rand(1)
                raise RuntimeError("deliberate test failure")

        model = BrokenModel()
        python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "deliberate test failure"):
            evaluate_probes(model)
        self.assertTrue(model.training)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))


if __name__ == "__main__":
    unittest.main()
