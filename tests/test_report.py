"""Portable report loading and safe embedding of user-controlled log text."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.report import generate_report


def embedded_data(path):
    document = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="training-data" type="application/json">(.*?)</script>', document, re.S)
    return document, json.loads(match.group(1))


class TrainingReportTests(unittest.TestCase):
    def test_empty_run_is_inspectable_without_optional_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = generate_report(directory)
            document, data = embedded_data(path)
            self.assertEqual(path.name, "report.html")
            self.assertEqual(data["games"], [])
            self.assertEqual(data["updates"], [])
            self.assertEqual(data["probes"], [])
            self.assertIn("No measurements yet.", document)
            self.assertNotIn("__TRAINING_DATA__", document)
            self.assertNotRegex(document, r'<script[^>]+src=')

    def test_malformed_tail_does_not_hide_completed_game(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game = {"game": 1, "termination": "move_limit", "truncated": True,
                    "outcome": "*", "result": 0, "plies": 2, "trace": [],
                    "moves": ["e2e4", "e7e5"]}
            (root / "games.jsonl").write_text(json.dumps(game) + '\n{"game":2', encoding="utf-8")
            (root / "config.json").write_text('{"unfinished":', encoding="utf-8")
            _, data = embedded_data(generate_report(root))
            self.assertEqual(data["games"], [game])
            self.assertEqual(len(data["warnings"]), 2)
            self.assertTrue(any("games.jsonl line 2" in warning for warning in data["warnings"]))

    def test_embedded_log_strings_cannot_terminate_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attack = '</script><script>alert("unsafe")</script>&<img src=x>'
            (root / "config.json").write_text(json.dumps({"config": {"label": attack}}), encoding="utf-8")
            document, data = embedded_data(generate_report(root))
            self.assertEqual(data["metadata"]["config"]["label"], attack)
            self.assertNotIn(attack, document)
            self.assertIn(r"\u003c/script\u003e", document)
            self.assertNotIn("innerHTML", document)

    def test_nonfinite_metric_record_is_skipped_with_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "updates.jsonl").write_text('{"step":1,"loss":NaN}\n', encoding="utf-8")
            _, data = embedded_data(generate_report(root))
            self.assertEqual(data["updates"], [])
            self.assertEqual(len(data["warnings"]), 1)

    def test_full_log_snapshot_preserves_metrics_and_game_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"schema_version": 1, "config": {"sims": 100, "max_moves": 200}}
            game = {"game": 1, "termination": "checkmate", "truncated": False,
                    "trace": [{"ply": 1, "fen": "example", "move": "g6g7", "value_prediction": .1}],
                    "outcome": "1-0", "result": 1, "plies": 1}
            update = {"game": 1, "step": 1, "policy_loss": 6.0, "policy_kl": 2.2, "gradient_norm": .5}
            probe = {"game": 1, "step": 1, "positions": [{"id": "mate", "passed": True}]}
            (root / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
            for name, record in (("games", game), ("updates", update), ("probes", probe)):
                (root / f"{name}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = root / "export" / "inspection.html"
            _, data = embedded_data(generate_report(root, output))
            self.assertEqual(data["metadata"], metadata)
            self.assertEqual(data["games"], [game])
            self.assertEqual(data["updates"], [update])
            self.assertEqual(data["probes"], [probe])
            self.assertEqual(data["warnings"], [])
            self.assertFalse(output.with_name(output.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
