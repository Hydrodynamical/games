"""Human imitation uses the selected player; outcome learning uses both sides."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
try:
    import chess as reference
    HAS_CHESS = hasattr(reference, "Board")
except ImportError:
    HAS_CHESS = False
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_CHESS and HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from scr import human_data, supervised


@unittest.skipUnless(HAS_CHESS and HAS_TORCH, "install requirements/human.txt for supervised tests")
class SupervisedStepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(38)

    def model(self):
        class SmallNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = torch.nn.Linear(2, 5)
                self.policy_head = torch.nn.Linear(5, 6)
                self.value_head = torch.nn.Linear(5, 1)

            def forward(self, states):
                hidden = torch.tanh(self.trunk(states))
                return self.policy_head(hidden), torch.tanh(self.value_head(hidden))

        return SmallNet()

    def batch(self):
        return {"states": torch.tensor([[.2, -.4], [1., .5], [-.7, .1], [.3, .8]]),
                "actions": torch.tensor([1, 3, 5, 2]),
                "values": torch.tensor([1., -1., 0., 1.]),
                "policy_mask": torch.tensor([True, False, True, False])}

    def test_mixed_update_matches_manual_player_policy_and_both_sides_value(self):
        model, batch = self.model(), self.batch()
        expected = deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        reference_optimizer = torch.optim.SGD(expected.parameters(), lr=.03)
        logits, values = expected(batch["states"])
        policy = F.cross_entropy(logits[[0, 2]], batch["actions"][[0, 2]])
        value = F.mse_loss(values.flatten(), batch["values"])
        reference_optimizer.zero_grad(set_to_none=True)
        (policy + .7 * value).backward()
        reference_optimizer.step()
        metrics = supervised.supervised_step(model, optimizer, batch, value_weight=.7)
        self.assertEqual(metrics["policy_sample_count"], 2)
        self.assertEqual(metrics["value_sample_count"], 4)
        self.assertAlmostEqual(metrics["policy_loss"], policy.item(), places=6)
        self.assertAlmostEqual(metrics["value_loss"], value.item(), places=6)
        self.assertAlmostEqual(metrics["zero_value_baseline_mse"], .75)
        for actual, reference_parameter in zip(model.parameters(), expected.parameters()):
            self.assertTrue(torch.equal(actual, reference_parameter))

    def test_opponent_moves_do_not_affect_policy_or_parameter_updates(self):
        model, batch = self.model(), self.batch()
        changed = deepcopy(model)
        changed_batch = {key: value.clone() for key, value in batch.items()}
        changed_batch["actions"][[1, 3]] = torch.tensor([4, 0])
        original_metrics = supervised.supervised_step(model, torch.optim.SGD(model.parameters(), lr=.02), batch)
        changed_metrics = supervised.supervised_step(changed, torch.optim.SGD(changed.parameters(), lr=.02), changed_batch)
        self.assertEqual(original_metrics, changed_metrics)
        for actual, expected in zip(model.parameters(), changed.parameters()):
            self.assertTrue(torch.equal(actual, expected))

    def test_opponent_outcome_still_affects_value_learning(self):
        model, batch = self.model(), self.batch()
        changed = deepcopy(model)
        changed_batch = {key: value.clone() for key, value in batch.items()}
        changed_batch["values"][1] = 1
        original_metrics = supervised.supervised_step(model, torch.optim.SGD(model.parameters(), lr=.02), batch)
        changed_metrics = supervised.supervised_step(changed, torch.optim.SGD(changed.parameters(), lr=.02), changed_batch)
        self.assertEqual(original_metrics["policy_loss"], changed_metrics["policy_loss"])
        self.assertNotEqual(original_metrics["value_loss"], changed_metrics["value_loss"])
        self.assertFalse(torch.equal(model.value_head.weight, changed.value_head.weight))

    def test_opponent_only_batch_has_no_policy_head_adam_drift(self):
        model, batch = self.model(), self.batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.1)
        supervised.supervised_step(model, optimizer, batch)
        before = {name: parameter.detach().clone() for name, parameter in model.policy_head.named_parameters()}
        optimizer_before = {name: deepcopy(optimizer.state[parameter])
                            for name, parameter in model.policy_head.named_parameters()}
        value_before = model.value_head.weight.detach().clone()
        batch["policy_mask"].fill_(False)
        metrics = supervised.supervised_step(model, optimizer, batch)
        self.assertIsNone(metrics["policy_loss"])
        self.assertIsNone(metrics["policy_top1_full_action"])
        self.assertEqual(metrics["policy_sample_count"], 0)
        self.assertEqual(metrics["value_sample_count"], 4)
        self.assertFalse(torch.equal(value_before, model.value_head.weight))
        for name, parameter in model.policy_head.named_parameters():
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(before[name], parameter))
            for key, value in optimizer_before[name].items():
                actual = optimizer.state[parameter][key]
                self.assertTrue(torch.equal(value, actual) if torch.is_tensor(value) else value == actual)

    def test_validation_uses_sample_denominators_for_uneven_batches(self):
        model, batch = self.model(), self.batch()
        # Different policy counts (one vs two) and value counts (one vs three)
        # expose an accidental mean-of-batch-means implementation.
        batch["policy_mask"] = torch.tensor([True, True, True, False])
        batches = [{key: value[:1] for key, value in batch.items()},
                   {key: value[1:] for key, value in batch.items()}]
        model.train()
        metrics = supervised.evaluate_batches(model, batches)
        self.assertTrue(model.training)
        with torch.no_grad():
            logits, values = model(batch["states"])
            selected = batch["policy_mask"]
            self.assertAlmostEqual(metrics["policy_nll_full_action"],
                                   F.cross_entropy(logits[selected], batch["actions"][selected]).item(), places=6)
            self.assertAlmostEqual(metrics["value_mse"], F.mse_loss(values.flatten(), batch["values"]).item(), places=6)
            self.assertAlmostEqual(metrics["value_mae"], (values.flatten() - batch["values"]).abs().mean().item(), places=6)
            self.assertAlmostEqual(metrics["policy_top1_full_action"],
                                   (logits[selected].argmax(1) == batch["actions"][selected]).float().mean().item(), places=6)
        self.assertEqual(metrics["policy_sample_count"], 3)
        self.assertEqual(metrics["value_sample_count"], 4)
        self.assertEqual(metrics["zero_value_baseline_mse"], .75)

    def test_empty_validation_is_unavailable_and_failures_restore_mode(self):
        model = self.model()
        model.eval()
        metrics = supervised.evaluate_batches(model, [])
        self.assertFalse(model.training)
        self.assertEqual(metrics["policy_sample_count"], 0)
        self.assertEqual(metrics["value_sample_count"], 0)
        self.assertIsNone(metrics["policy_nll_full_action"])
        self.assertIsNone(metrics["value_mse"])
        model.train()
        batch = self.batch()
        batch["states"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            supervised.evaluate_batches(model, [batch])
        self.assertTrue(model.training)


@unittest.skipUnless(HAS_CHESS and HAS_TORCH, "install requirements/human.txt for supervised tests")
class SupervisedRunTests(unittest.TestCase):
    def prepare(self, directory):
        source, prepared = directory / "games.jsonl", directory / "prepared"
        records = []
        for index in range(10):
            white, black = ("Hikaru", "Other") if index % 2 == 0 else ("Other", "Hikaru")
            result = ("1-0", "0-1", "1/2-1/2")[index % 3]
            records.append({"game_id": f"test-{index}", "white": {"username": white}, "black": {"username": black},
                            "target_color": "w" if index % 2 == 0 else "b", "result": result,
                            "end_time": int(datetime(2026, 1, index + 1, tzinfo=timezone.utc).timestamp()),
                            "rules": "chess", "termination": "recorded outcome", "time_class": "blitz",
                            "time_control": "180", "rated": True,
                            "pgn": f'[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n\n1. e4 e5 2. Nf3 Nc6 {result}'})
        source.write_text("".join(json.dumps(record) + "\n" for record in records))
        human_data.prepare_dataset(source, prepared, shard_size=9)
        return prepared

    @staticmethod
    def tiny_network(hidden=8):
        class TinyChessNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = torch.nn.Linear(20, hidden)
                self.policy_head = torch.nn.Linear(hidden, 4480)
                self.value_head = torch.nn.Linear(hidden, 1)

            def forward(self, states):
                hidden = torch.tanh(self.trunk(states.mean(dim=(2, 3))))
                return self.policy_head(hidden), torch.tanh(self.value_head(hidden))

        return TinyChessNet()

    def run_training(self, dataset, output, **kwargs):
        return supervised.train_supervised(dataset, output, batch_size=5, hidden=8, threads=1,
                                           seed=12, device="cpu", **kwargs)

    @staticmethod
    def validation_sequence(*nlls):
        return [{"policy_sample_count": 2, "value_sample_count": 4,
                 "policy_nll_full_action": nll, "value_mse": .5} for nll in nlls]

    def assert_checkpoint_equal(self, first, second):
        first_state = torch.load(first, weights_only=True)["state_dict"]
        second_state = torch.load(second, weights_only=True)["state_dict"]
        self.assertEqual(first_state.keys(), second_state.keys())
        for name, tensor in first_state.items():
            self.assertTrue(torch.equal(tensor, second_state[name]), name)

    def test_runner_checkpoints_resume_identical_to_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(supervised, "PolicyValueNet", self.tiny_network):
            root = Path(temporary)
            dataset = self.prepare(root)
            full = self.run_training(dataset, root / "full", epochs=2)
            first = self.run_training(dataset, root / "first", epochs=1)
            resumed = self.run_training(dataset, root / "resumed", epochs=1,
                                        resume=root / "first/checkpoints/epoch_1.pt")
            self.assertEqual(full["status"], "completed")
            self.assertEqual(resumed["completed_epoch"], 2)
            self.assertEqual(full["optimizer_steps"], resumed["optimizer_steps"])
            self.assertEqual(full["last_validation"], resumed["last_validation"])
            self.assertEqual(first["epochs"][0]["policy_samples"], 16)
            self.assertEqual(first["epochs"][0]["value_samples"], 32)
            actual = torch.load(root / "resumed/checkpoints/epoch_2.pt", weights_only=True)
            expected = torch.load(root / "full/checkpoints/epoch_2.pt", weights_only=True)
            for name, tensor in expected["state_dict"].items():
                self.assertTrue(torch.equal(tensor, actual["state_dict"][name]), name)
            self.assertTrue(torch.equal(expected["torch_rng_state"], actual["torch_rng_state"]))
            self.assertEqual(expected["python_rng_state"], actual["python_rng_state"])
            self.assertTrue((root / "first/source/chess/scr/supervised.py").is_file())
            self.assertTrue((root / "first/checkpoints/best.pt").is_file())
            config = json.loads((root / "first/config.json").read_text())
            self.assertEqual(config["dataset_manifest_sha256"], supervised.sha256(dataset / "manifest.json"))

    def test_resume_rejects_changed_dataset_and_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(supervised, "PolicyValueNet", self.tiny_network):
            root = Path(temporary)
            dataset = self.prepare(root)
            self.run_training(dataset, root / "first", epochs=1)
            with self.assertRaisesRegex(ValueError, "empty or new"):
                self.run_training(dataset, root / "first", epochs=1)
            manifest_path = dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["test_identity_change"] = True
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "configuration or dataset"):
                self.run_training(dataset, root / "mismatch", epochs=1,
                                  resume=root / "first/checkpoints/epoch_1.pt")

    def test_worse_epoch_does_not_displace_initial_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(supervised, "PolicyValueNet", self.tiny_network):
            root = Path(temporary)
            dataset = self.prepare(root)
            base = {"policy_sample_count": 2, "value_sample_count": 4,
                    "policy_nll_full_action": 1., "value_mse": .5}
            with patch.object(supervised, "evaluate_batches", side_effect=[base, dict(base, policy_nll_full_action=2.)]):
                summary = self.run_training(dataset, root / "run", epochs=1)
            initial = torch.load(root / "run/checkpoints/initial.pt", weights_only=True)["state_dict"]
            best = torch.load(summary["best_checkpoint"], weights_only=True)["state_dict"]
            final = torch.load(summary["final_checkpoint"], weights_only=True)["state_dict"]
            self.assertTrue(all(torch.equal(initial[key], best[key]) for key in initial))
            self.assertTrue(any(not torch.equal(initial[key], final[key]) for key in initial))

    def test_worsening_validation_stops_cleanly_and_keeps_earlier_best(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(supervised, "PolicyValueNet", self.tiny_network):
            root = Path(temporary)
            dataset = self.prepare(root)
            with patch.object(supervised, "evaluate_batches", side_effect=self.validation_sequence(1., .8, .9, 1.)):
                summary = self.run_training(dataset, root / "run", epochs=8, patience=2)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["epochs_requested"], 8)
            self.assertEqual(summary["epochs_completed"], 3)
            self.assertEqual(summary["completed_epoch"], 3)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["best_validation_nll"], .8)
            self.assertEqual(summary["epochs_without_improvement"], 2)
            self.assertTrue(summary["stopped_early"])
            self.assertEqual(summary["stop_reason"], "patience_exhausted")
            self.assert_checkpoint_equal(summary["best_checkpoint"], root / "run/checkpoints/epoch_1.pt")
            self.assert_checkpoint_equal(summary["final_checkpoint"], root / "run/checkpoints/epoch_3.pt")
            self.assertFalse((root / "run/checkpoints/epoch_4.pt").exists())

    def test_min_delta_patience_is_separate_from_absolute_best(self):
        # Small decreases remain eligible for best.pt; their cumulative decrease
        # resets patience once it exceeds min_delta from the last reset.
        cases = [("small", [1., .96, .94], 2, 0., .94),
                 ("cumulative", [1., .96, .89, .88, .87], 4, .89, .87)]
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(supervised, "PolicyValueNet", self.tiny_network):
            root = Path(temporary)
            dataset = self.prepare(root)
            for name, nlls, completed, reference, best in cases:
                with self.subTest(name=name), patch.object(supervised, "evaluate_batches",
                                                          side_effect=self.validation_sequence(*nlls)):
                    summary = self.run_training(dataset, root / name, epochs=8, patience=2, min_delta=.1)
                self.assertEqual(summary["completed_epoch"], completed)
                self.assertEqual(summary["best_epoch"], completed)
                self.assertEqual(summary["best_validation_nll"], best)
                self.assertEqual(summary["patience_reference_nll"], reference or 1.)
                self.assertEqual(summary["epochs_without_improvement"], 2)
                self.assertTrue(summary["stopped_early"])
                self.assert_checkpoint_equal(summary["best_checkpoint"], summary["final_checkpoint"])

    def test_resume_preserves_best_weights_and_patience_and_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(supervised, "PolicyValueNet", self.tiny_network):
            root = Path(temporary)
            dataset = self.prepare(root)
            with patch.object(supervised, "evaluate_batches", side_effect=self.validation_sequence(1., .8, .9, 1., 1.1)):
                full = self.run_training(dataset, root / "full", epochs=8, patience=3)
            with patch.object(supervised, "evaluate_batches", side_effect=self.validation_sequence(1., .8, .9)):
                self.run_training(dataset, root / "first", epochs=2, patience=3)
            with patch.object(supervised, "evaluate_batches", side_effect=self.validation_sequence(.9, 1., 1.1)):
                resumed = self.run_training(dataset, root / "resumed", epochs=6, patience=3,
                                            resume=root / "first/checkpoints/epoch_2.pt")
            for key in ("completed_epoch", "optimizer_steps", "best_epoch", "best_validation_nll",
                        "epochs_without_improvement", "patience_reference_nll", "stopped_early", "stop_reason"):
                self.assertEqual(full[key], resumed[key], key)
            self.assertEqual(resumed["epochs_completed"], 2)
            self.assertEqual(resumed["best_epoch"], 1)
            self.assertEqual(resumed["completed_epoch"], 4)
            self.assert_checkpoint_equal(full["final_checkpoint"], resumed["final_checkpoint"])
            self.assert_checkpoint_equal(full["best_checkpoint"], resumed["best_checkpoint"])
            self.assert_checkpoint_equal(resumed["best_checkpoint"], root / "first/checkpoints/epoch_1.pt")
            config = json.loads((root / "resumed/config.json").read_text())
            self.assertEqual(config["inherited_best_epoch"], 1)
            # An exhausted checkpoint resumes without training extra epochs.
            with patch.object(supervised, "evaluate_batches", side_effect=self.validation_sequence(1.1)):
                stopped = self.run_training(dataset, root / "stopped", epochs=2, patience=3,
                                            resume=root / "resumed/checkpoints/epoch_4.pt")
            self.assertEqual(stopped["epochs_completed"], 0)
            self.assertTrue(stopped["stopped_early"])
            self.assert_checkpoint_equal(stopped["final_checkpoint"], resumed["final_checkpoint"])

    def test_full_train_evaluation_uses_frozen_initial_and_epoch_weights(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(supervised, "PolicyValueNet", self.tiny_network):
            root = Path(temporary)
            dataset = self.prepare(root)
            summary = self.run_training(dataset, root / "run", epochs=1, evaluate_train=True)
            history = json.loads((root / "run/validation.json").read_text())
            for entry, checkpoint in zip(history, ("initial.pt", "epoch_1.pt")):
                model = self.tiny_network()
                model.load_state_dict(torch.load(root / "run/checkpoints" / checkpoint, weights_only=True)["state_dict"])
                expected = supervised.evaluate_batches(model, human_data.iter_batches(dataset, "train", 5))
                self.assertEqual(entry["training_evaluation"], expected)
                self.assertEqual(expected["policy_sample_count"], 16)
                self.assertEqual(expected["value_sample_count"], 32)
            self.assertEqual(summary["initial_training_evaluation"], history[0]["training_evaluation"])
            self.assertEqual(summary["last_training_evaluation"], history[1]["training_evaluation"])
            self.assertEqual(summary["epochs"][0]["training_evaluation"], history[1]["training_evaluation"])

    def test_old_checkpoint_falls_back_to_starting_model_and_disabled_patience_continues(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch.object(supervised, "PolicyValueNet", self.tiny_network):
            root = Path(temporary)
            dataset = self.prepare(root)
            with patch.object(supervised, "evaluate_batches", side_effect=self.validation_sequence(1., 2.)):
                self.run_training(dataset, root / "first", epochs=1)
            old = torch.load(root / "first/checkpoints/epoch_1.pt", weights_only=True)
            del old["best_state_dict"]
            for key in ("best_epoch", "patience", "min_delta", "patience_reference_nll", "epochs_without_improvement"):
                del old["training_state"][key]
            torch.save(old, root / "old.pt")
            with patch.object(supervised, "evaluate_batches", side_effect=self.validation_sequence(2., 3., 4.)):
                summary = self.run_training(dataset, root / "resumed", epochs=2, resume=root / "old.pt")
            self.assertEqual(summary["completed_epoch"], 3)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["best_validation_nll"], 2.)
            self.assertFalse(summary["stopped_early"])
            self.assertEqual(summary["stop_reason"], "epochs_completed")
            self.assert_checkpoint_equal(summary["best_checkpoint"], root / "old.pt")

    def test_invalid_early_stopping_parameters_fail_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kwargs in ({"patience": -1}, {"patience": .5}, {"patience": True},
                           {"min_delta": -1.}, {"min_delta": float("nan")}, {"min_delta": float("inf")}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.run_training(root / "unused", root / "run", **kwargs)
                self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
