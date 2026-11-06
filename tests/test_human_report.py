"""Human-training reports retain metric populations and tolerate partial runs."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.human_report import generate_report


def read_report(path):
    document = path.read_text()
    match = re.search(r'<script id="human-data" type="application/json">(.*?)</script>', document, re.S)
    return document, json.loads(match.group(1))


DOM_HARNESS = r'''
const fs=require("fs"),vm=require("vm"),assert=require("assert");
const html=fs.readFileSync(process.argv[1],"utf8");
const payload=html.match(/<script id="human-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Element{constructor(){this.children=[];this.listeners={};this.style={setProperty(){}};this.classList={add(){},remove(){}};this.textContent="";this.value="";}append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;}setAttribute(){}addEventListener(name,fn){this.listeners[name]=fn;}click(){this.listeners.click?.();}}
const elements={};const document={getElementById(id){return elements[id]??=(new Element());},createElement(){return new Element();},createElementNS(){return new Element();},createTextNode(value){const e=new Element();e.textContent=value;return e;}};
document.getElementById("human-data").textContent=payload;
vm.runInNewContext(script,{document,Blob,URL,setTimeout,console});
assert.equal(elements.board.children.length,64);
assert.equal(elements["position-select"].children.length,2);
assert.match(elements["position-label"].textContent,/ply 1/);
elements.next.listeners.click();assert.match(elements["position-label"].textContent,/ply 3/);
elements.previous.listeners.click();assert.match(elements["position-label"].textContent,/ply 1/);
const flatten=e=>e.textContent+e.children.map(flatten).join(" ");
assert.match(flatten(elements["matches-table"]),/75.0% \(2 games\)/);
assert.match(flatten(elements["comparison-table"]),/40.0%/);
console.log("Human report DOM checks passed");
'''


class HumanReportTests(unittest.TestCase):
    @staticmethod
    def fixture(directory):
        root = Path(directory)
        (root / "config.json").write_text(json.dumps({"epochs": 1, "policy_supervision": "Hikaru only"}))
        (root / "dataset_manifest.json").write_text(json.dumps({
            "accepted_games": 5000, "split": {"strategy": "chronological_whole_utc_date_groups"},
            "splits": {"train": {"games": 4000, "positions": 200000, "policy_positions": 100000,
                                  "first_end_time": 1000, "last_end_time": 2000}}}))
        (root / "updates.jsonl").write_text(json.dumps({"step": 1, "policy_loss": 7., "value_loss": .8}) + "\n")
        (root / "validation.json").write_text(json.dumps([
            {"epoch": 0, "step": 0, "validation": {"policy_nll_full_action": 8., "value_mse": 1.}},
            {"epoch": 1, "step": 1, "validation": {"policy_nll_full_action": 6., "value_mse": .8}}]))
        final = root / "eval-final"
        final.mkdir()
        row = {"game_id": "game", "ply": 1, "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
               "human_move": "e2e4", "raw_move": "e2e4", "search_move": "d2d4", "result": "1-0",
               "raw_top5": ["e2e4"], "search_top5": ["d2d4"], "human_move_probability": .4,
               "human_move_search_probability": .2, "value": .1, "target_value": 1., "phase": "opening",
               "raw_seconds": .001, "search_seconds": .1}
        rows = [row, dict(row, ply=3, human_move="g1f3")]
        (final / "positions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (final / "summary.json").write_text(json.dumps({
            "status": "completed", "heldout": {"all": {"positions": 2, "raw_top1": .4, "search_top1": .2,
            "raw_top5": .6, "search_top5": .5, "raw_legal_nll": 2., "move_change_fraction": .8}},
            "tactical_probes": {"raw_passes": 1, "search_passes": 3, "scored_positions": 4}}))
        matches = [{"mcts_score": 1., "termination": "checkmate"},
                   {"mcts_score": .5, "termination": "stalemate"},
                   {"mcts_score": 0., "termination": "move_limit", "truncated": True}]
        (final / "matches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in matches))
        return root

    def test_empty_and_partial_runs_are_inspectable(self):
        with tempfile.TemporaryDirectory() as directory:
            document, data = read_report(generate_report(directory))
            self.assertEqual(data["updates"], [])
            self.assertEqual(data["validation"], [])
            self.assertIsNone(data["initial_final_samples_match"])
            self.assertIn("No measurements yet.", document)
            self.assertNotRegex(document, r'<script[^>]+src=')
            root = Path(directory)
            (root / "updates.jsonl").write_text('{"step":1,"policy_loss":4}\n{"step":2')
            (root / "validation.json").write_text('{"wrong":"type"}')
            _, data = read_report(generate_report(root))
            self.assertEqual(len(data["updates"]), 1)
            self.assertEqual(len(data["warnings"]), 2)

    def test_metrics_keep_distinct_populations_and_cutoff_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            document, data = read_report(generate_report(root))
            totals = data["evaluations"]["final"]["match_totals"]
            self.assertEqual(totals["completed_games"], 2)
            self.assertEqual(totals["unresolved_cutoffs"], 1)
            self.assertEqual(totals["actual_draws"], 1)
            self.assertEqual(totals["policy_wins"], 0)
            self.assertEqual(totals["mcts_score_on_completed_games"], .75)
            self.assertEqual(data["validation"][-1]["validation"]["policy_nll_full_action"], 6.)
            self.assertEqual(data["evaluations"]["final"]["summary"]["heldout"]["all"]["raw_legal_nll"], 2.)
            self.assertIn("not an equal-time comparison", document)
            self.assertIn("all 4,480 action outputs", document)

    def test_untrusted_strings_cannot_escape_the_data_script(self):
        with tempfile.TemporaryDirectory() as directory:
            attack = '</script><script>alert("x")</script>&<img src=x>'
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"label": attack}))
            document, data = read_report(generate_report(root))
            self.assertEqual(data["config"]["label"], attack)
            self.assertNotIn(attack, document)
            self.assertIn(r"\u003c/script\u003e", document)
            self.assertNotIn("innerHTML", document)

    def test_initial_final_sample_mismatch_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            initial = root / "eval-initial"
            initial.mkdir()
            (initial / "positions.jsonl").write_text('{"game_id":"different","ply":1}\n')
            _, data = read_report(generate_report(root))
            self.assertFalse(data["initial_final_samples_match"])
            self.assertTrue(any("samples differ" in warning for warning in data["warnings"]))

    def test_nonfinite_records_are_rejected_and_custom_output_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "updates.jsonl").write_text('{"step":1,"value_loss":Infinity}\n')
            output = root / "export" / "report.html"
            _, data = read_report(generate_report(root, output))
            self.assertEqual(data["updates"], [])
            self.assertEqual(len(data["warnings"]), 1)
            self.assertFalse(list(output.parent.glob("*.tmp")))

    @unittest.skipUnless(shutil.which("node"), "Node is optional for report DOM regression")
    def test_board_controls_and_match_score_render(self):
        with tempfile.TemporaryDirectory() as directory:
            path = generate_report(self.fixture(directory))
            subprocess.run([shutil.which("node"), "-e", DOM_HARNESS, str(path)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
