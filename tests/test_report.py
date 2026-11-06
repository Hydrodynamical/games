"""Portable report loading and safe embedding of user-controlled log text."""
import json
from pathlib import Path
import re
import shutil
import subprocess
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
    def test_masked_batch_has_no_value_measurement_but_retains_policy_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"config": {"cutoff_value_policy": "exclude"}}
            update = {"step": 1, "batch_size": 4, "policy_sample_count": 4,
                      "value_sample_count": 0, "value_sample_fraction": 0,
                      "value_metrics_available": False, "loss": 2.0, "policy_loss": 2.0,
                      "value_loss": 0.0, "zero_value_baseline_mse": None,
                      "cutoff_target_fraction": 1.0, "target_win_fraction": 0.0,
                      "target_loss_fraction": 0.0}
            (root / "config.json").write_text(json.dumps(metadata))
            (root / "updates.jsonl").write_text(json.dumps(update) + "\n")
            _, data = embedded_data(generate_report(root))
            self.assertEqual(data["updates"], [update])
            display = data["value_updates"][0]
            self.assertEqual(display["policy_loss"], 2.0)
            self.assertIsNone(display["value_loss"])
            self.assertIsNone(display["zero_value_baseline_mse"])
            self.assertIsNone(display["target_value_mean"])
            self.assertEqual(display["policy_only_fraction"], 1.0)

    def test_baseline_uses_eligible_denominator_and_legacy_logs_use_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            update = {"step": 1, "batch_size": 8, "cutoff_target_fraction": .5,
                      "target_win_fraction": .25, "target_loss_fraction": .25,
                      "value_loss": .9}
            (root / "updates.jsonl").write_text(json.dumps(update) + "\n")
            # Missing masking configuration identifies the historical baseline.
            _, legacy = embedded_data(generate_report(root))
            self.assertEqual(legacy["value_updates"][0]["zero_value_baseline_mse"], .5)
            self.assertEqual(legacy["value_updates"][0]["value_sample_count"], 8)
            (root / "config.json").write_text(json.dumps({"config": {"cutoff_value_policy": "exclude"}}))
            _, masked = embedded_data(generate_report(root))
            self.assertEqual(masked["value_updates"][0]["zero_value_baseline_mse"], 1.0)
            self.assertEqual(masked["value_updates"][0]["value_sample_count"], 4)
            self.assertEqual(masked["value_updates"][0]["value_sample_fraction"], .5)
            # An explicitly logged zero baseline (all eligible rows are actual draws)
            # takes precedence over any origin-fraction approximation.
            update["zero_value_baseline_mse"] = 0.0
            (root / "updates.jsonl").write_text(json.dumps(update) + "\n")
            _, explicit = embedded_data(generate_report(root))
            self.assertEqual(explicit["value_updates"][0]["zero_value_baseline_mse"], 0.0)

    @unittest.skipUnless(shutil.which("node"), "Node is optional for report DOM regression")
    def test_game_inspector_displays_policy_only_instead_of_zero_value_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"config": {"cutoff_value_policy": "exclude"}}
            trace = [{"ply": 1, "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                      "player": "w", "move": "e1e2", "value_prediction": .2,
                      "top_moves": []}]
            games = [
                {"game": 1, "termination": "move_limit", "truncated": True,
                 "value_target_available": False, "outcome": "*", "result": 0,
                 "plies": 1, "trace": trace, "moves": ["e1e2"]},
                {"game": 2, "termination": "stalemate", "truncated": False,
                 "value_target_available": True, "outcome": "1/2-1/2", "result": 0,
                 "plies": 1, "trace": trace, "moves": ["e1e2"]},
            ]
            (root / "config.json").write_text(json.dumps(metadata))
            (root / "games.jsonl").write_text("".join(json.dumps(game) + "\n" for game in games))
            document, data = embedded_data(generate_report(root))
            script = re.findall(r"<script>(.*?)</script>", document, re.S)[0]
            harness = r'''
const fs=require("fs"),vm=require("vm"),assert=require("assert");
const input=JSON.parse(fs.readFileSync(0,"utf8"));
class Element{
 constructor(tag){this.tag=tag;this.children=[];this._text="";this.style={};this.handlers={};this.value="";this.classList={add(){},remove(){}};}
 set textContent(text){this._text=String(text);this.children=[];}
 get textContent(){return this._text+this.children.map(child=>child.textContent).join("");}
 append(...children){this.children.push(...children);if(this.tag==="select"&&this.value===""&&children[0])this.value=children[0].value;}
 replaceChildren(...children){this.children=children;this._text="";}
 setAttribute(name,value){this[name]=value;}
 addEventListener(name,fn){this.handlers[name]=fn;}
 getBoundingClientRect(){return {left:0,width:540};}
}
const elements=new Map();
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element(id.endsWith("select")?"select":"div"));return elements.get(id);},createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>{const el=new Element("text");el.textContent=text;return el;}};
document.getElementById("training-data").textContent=JSON.stringify(input.data);
const context=vm.createContext({document,console,URL,Blob,setTimeout});
vm.runInContext(input.script,context);
assert(elements.get("trace-metrics").textContent.includes("Excluded · policy only"));
assert(!elements.get("trace-metrics").textContent.includes("Cutoff training target"));
assert(elements.get("cutoff-policy-note").textContent.includes("excluded from value loss"));
assert(elements.get("outcome-note").textContent.includes("1 positions are policy-only"));
assert.strictEqual(elements.get("board").children.length,64);
vm.runInContext("gameIndex=1;showGame();",context);
assert(elements.get("trace-metrics").textContent.includes("Final outcome target0"));
assert(!elements.get("trace-metrics").textContent.includes("Excluded"));
vm.runInContext("gameIndex=0;positionIndex=1;showGame(false);",context);
assert(elements.get("trace-extra").textContent.includes("contributes policy targets only"));
console.log("masked report DOM checks passed");
'''
            completed = subprocess.run([shutil.which("node"), "-e", harness], input=json.dumps({"data": data, "script": script}),
                                       text=True, capture_output=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)

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
