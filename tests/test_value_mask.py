"""Unfinished positions teach policy while contributing no value supervision."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from scr.train import train_step


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional training dependency")
class ValueMaskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def model(self):
        class SmallNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = torch.nn.Linear(2, 4)
                self.policy_head = torch.nn.Linear(4, 3)
                self.value_head = torch.nn.Linear(4, 1)

            def forward(self, states):
                hidden = torch.tanh(self.trunk(states))
                return self.policy_head(hidden), torch.tanh(self.value_head(hidden))

        return SmallNet()

    def batch(self):
        return [
            (torch.tensor([.2, -.4]), torch.tensor([.75, .25, 0.]), 1.),
            (torch.tensor([1., .5]), torch.tensor([0., 0., 1.]), 0.),
            (torch.tensor([-.7, .1]), torch.tensor([.1, .8, .1]), -1.),
        ]

    def test_mixed_mask_matches_manual_policy_all_value_selected_update(self):
        model, batch = self.model(), self.batch()
        reference = deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
        states = torch.stack([row[0] for row in batch])
        policies = torch.stack([row[1] for row in batch])
        targets = torch.tensor([row[2] for row in batch])
        logits, values = reference(states)
        expected_policy = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        expected_value = F.mse_loss(values.squeeze(1)[[0, 2]], targets[[0, 2]])
        reference_optimizer.zero_grad(set_to_none=True)
        (expected_policy + expected_value).backward()
        reference_optimizer.step()

        metrics = train_step(model, optimizer, batch, "cpu", value_mask=[True, False, True])
        self.assertAlmostEqual(metrics["policy_loss"], expected_policy.detach().item(), places=6)
        self.assertAlmostEqual(metrics["value_loss"], expected_value.detach().item(), places=6)
        self.assertEqual(metrics["policy_sample_count"], 3)
        self.assertEqual(metrics["value_sample_count"], 2)
        self.assertAlmostEqual(metrics["value_sample_fraction"], 2 / 3)
        self.assertTrue(metrics["value_metrics_available"])
        self.assertEqual(metrics["target_value_mean"], 0)
        self.assertEqual(metrics["target_value_std"], 1)
        self.assertEqual(metrics["zero_value_baseline_mse"], 1)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(actual, expected))

    def test_all_masked_still_trains_policy_without_adam_value_head_drift(self):
        model, batch = self.model(), self.batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.1)
        train_step(model, optimizer, batch, "cpu")  # Prime momentum and weight decay state.
        before_value = {name: parameter.detach().clone() for name, parameter in model.value_head.named_parameters()}
        before_state = {name: {key: value.clone() if torch.is_tensor(value) else value
                              for key, value in optimizer.state[parameter].items()}
                        for name, parameter in model.value_head.named_parameters()}
        before_trunk = model.trunk.weight.detach().clone()
        before_policy = model.policy_head.weight.detach().clone()

        for _ in range(2):
            metrics = train_step(model, optimizer, batch, "cpu", value_mask=[False] * len(batch))
            self.assertEqual(metrics["value_loss"], 0)
            self.assertEqual(metrics["loss"], metrics["policy_loss"])
            self.assertEqual(metrics["policy_sample_count"], 3)
            self.assertEqual(metrics["value_sample_count"], 0)
            self.assertFalse(metrics["value_metrics_available"])
            for key in ("value_mae", "predicted_value_mean", "predicted_value_std", "target_value_mean",
                        "target_value_std", "zero_value_baseline_mse"):
                self.assertIsNone(metrics[key])
            self.assertTrue(all(value is None or math.isfinite(value) for value in metrics.values()))
            json.dumps(metrics, allow_nan=False)
            for name, parameter in model.value_head.named_parameters():
                self.assertIsNone(parameter.grad)
                self.assertTrue(torch.equal(parameter, before_value[name]))
                for key, original in before_state[name].items():
                    actual = optimizer.state[parameter][key]
                    self.assertTrue(torch.equal(actual, original) if torch.is_tensor(original) else actual == original)
        self.assertFalse(torch.equal(model.trunk.weight, before_trunk))
        self.assertFalse(torch.equal(model.policy_head.weight, before_policy))

    def test_all_eligible_matches_legacy_loss_and_update_exactly(self):
        original, batch = self.model(), self.batch()
        explicit = deepcopy(original)
        legacy_metrics = train_step(original, torch.optim.Adam(original.parameters(), lr=.001), batch, "cpu")
        masked_metrics = train_step(explicit, torch.optim.Adam(explicit.parameters(), lr=.001), batch, "cpu",
                                    value_mask=torch.ones(len(batch), dtype=torch.bool))
        self.assertEqual(legacy_metrics, masked_metrics)
        for actual, expected in zip(original.parameters(), explicit.parameters()):
            self.assertTrue(torch.equal(actual, expected))

    def test_real_draw_zero_remains_value_supervised(self):
        model = self.model()
        batch = [self.batch()[1]]
        metrics = train_step(model, torch.optim.SGD(model.parameters(), lr=.03), batch, "cpu", value_mask=[1.])
        self.assertEqual(metrics["value_sample_count"], 1)
        self.assertTrue(metrics["value_metrics_available"])
        self.assertEqual(metrics["target_value_mean"], 0)
        self.assertEqual(metrics["zero_value_baseline_mse"], 0)
        self.assertGreater(metrics["value_loss"], 0)
        self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in model.value_head.parameters()), 0)

    def test_excluded_target_cannot_change_loss_or_gradients(self):
        model, batch = self.model(), self.batch()
        changed = deepcopy(model)
        changed_batch = list(batch)
        changed_batch[1] = (*batch[1][:2], 1.)
        before = train_step(model, torch.optim.SGD(model.parameters(), lr=.02), batch, "cpu", value_mask=[1, 0, 1])
        after = train_step(changed, torch.optim.SGD(changed.parameters(), lr=.02), changed_batch, "cpu", value_mask=[1, 0, 1])
        self.assertEqual(before, after)
        for actual, expected in zip(model.parameters(), changed.parameters()):
            self.assertTrue(torch.equal(actual, expected))

    def test_bad_masks_fail_before_parameter_update(self):
        batch = self.batch()
        for mask in (True, [], [True], [[1], [0], [1]], [1, -1, 1], [1, .5, 1],
                     [1, float("nan"), 1], [1, float("inf"), 1], [1, 1j, 1], ["yes"] * 3):
            with self.subTest(mask=mask):
                model = self.model()
                before = [parameter.detach().clone() for parameter in model.parameters()]
                with self.assertRaisesRegex(ValueError, "value_mask"):
                    train_step(model, torch.optim.SGD(model.parameters(), lr=.03), batch, "cpu", value_mask=mask)
                for actual, expected in zip(model.parameters(), before):
                    self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
