"""Offline self-play reports preserve outcome populations and safe histories."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.chess_engine import GameState
from scr.selfplay_report import generate_report


def read_report(path):
    document = path.read_text()
    payload = re.search(r'<script id="selfplay-data" type="application/json">(.*?)</script>', document, re.S)
    return document, json.loads(payload.group(1))


DOM_HARNESS = r'''
const fs=require("fs"),vm=require("vm"),assert=require("assert"),html=fs.readFileSync(process.argv[1],"utf8");
const payload=html.match(/<script id="selfplay-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Element{constructor(){this.children=[];this.listeners={};this.style={setProperty(){}};this.classList={add(){},remove(){}};this.textContent="";this.value="";}append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;}setAttribute(){}addEventListener(name,fn){this.listeners[name]=fn;}click(){this.listeners.click?.();}}
const elements={},document={getElementById(id){return elements[id]??=new Element();},createElement(){return new Element();},createElementNS(){return new Element();}};
document.getElementById("selfplay-data").textContent=payload;
vm.runInNewContext(script,{document,Blob,URL,setTimeout,console});
const flatten=e=>e.textContent+e.children.map(flatten).join(" "),data=JSON.parse(payload);
assert.match(elements["selection-note"].textContent,/preselected final checkpoint/);
assert.equal(elements["training-report"].href,data.training_report_href);
if(data.matches.length){
 assert.equal(elements.board.children.length,64);assert.equal(elements["match-select"].children.length,3);
 assert.match(elements["ply-label"].textContent,/Ply 0 \/ 4/);
 elements.next.listeners.click();assert.match(elements["ply-label"].textContent,/Ply 1 \/ 4/);
 elements.last.listeners.click();assert.match(elements["ply-label"].textContent,/Ply 4 \/ 4/);
 elements.previous.listeners.click();assert.match(elements["ply-label"].textContent,/Ply 3 \/ 4/);
 elements["match-select"].value="1";elements["match-select"].listeners.change();assert.match(elements["ply-label"].textContent,/Ply 0 \/ 0/);
 assert.match(flatten(elements["match-table"]),/75.0% \(2 games\)/);
 assert.match(elements["bounds-note"].textContent,/50.0% to 83.3%/);
 assert.match(flatten(elements["value-breakdown"]),/Decisive games/);
 assert.match(flatten(elements["value-chart"]),/Always zero/);
 assert.match(flatten(elements["retention-table"]),/6.2500/);
}else{assert.equal(elements.previous.disabled,true);assert.match(flatten(elements["value-table"]),/No measurements yet/);}
console.log("Self-play report DOM checks passed");
'''


class SelfplayReportTests(unittest.TestCase):
    @staticmethod
    def metrics(positions, games, baseline, candidate, zero):
        return {"positions": positions, "games": games,
                "baseline": {"mse": baseline, "mae": baseline, "mean_absolute_prediction": .3},
                "candidate": {"mse": candidate, "mae": candidate, "mean_absolute_prediction": .2},
                "zero": {"mse": zero, "mae": zero},
                "candidate_minus_baseline_mse": candidate - baseline,
                "candidate_minus_baseline_mae": candidate - baseline}

    @classmethod
    def fixture(cls, directory):
        root = Path(directory)
        (root / "config.json").write_text(json.dumps({"config": {"iterations": 100, "sims": 64,
            "max_moves": 400, "updates": 8, "cutoff_value_policy": "draw"}}))
        (root / "summary.json").write_text(json.dumps({"status": "completed", "games_completed": 100,
            "optimizer_steps": 792, "move_limit_games": 8, "move_limit_fraction": .08,
            "actual_draws": 30, "decisive_games": 62, "cutoff_target_fraction": .12}))
        comparison = root / "comparison"
        comparison.mkdir()
        values = {"games": 4, "completed_games": 3, "excluded_unknown_outcome_games": 1,
                  "excluded_unknown_outcome_positions": 400,
                  "all": cls.metrics(12, 3, .7, .6, .75),
                  "by_game_result": {"1-0": cls.metrics(3, 1, .8, .7, 1.),
                                     "0-1": cls.metrics(6, 1, 1., .9, 1.),
                                     "1/2-1/2": cls.metrics(3, 1, .1, .05, 0.)},
                  "bootstrap_game_level": {"samples": 2000, "scored_games": 3,
                      "candidate_minus_baseline_mse": [-.2, .01], "candidate_minus_baseline_mae": [-.15, .02]}}
        (comparison / "summary.json").write_text(json.dumps({"status": "completed", "values": values,
            "config": {"sims": 64, "max_plies_after_opening": 400, "match_pairs": 10}}))
        state = GameState()
        initial_fen = state.to_fen()
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
        for move in moves:
            state.push_uci(move)
        draw_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        cutoff = GameState()
        cutoff.push_uci("e2e4")
        matches = [
            {"game_id": "mate", "opening": "test", "candidate_color": "b", "initial_fen": initial_fen,
             "moves": moves, "final_fen": state.to_fen(), "candidate_score": 1., "outcome": "0-1",
             "termination": "checkmate", "truncated": False},
            {"game_id": "draw", "candidate_color": "w", "initial_fen": draw_fen, "moves": [],
             "final_fen": draw_fen, "candidate_score": .5, "outcome": "1/2-1/2",
             "termination": "insufficient_material", "truncated": False},
            {"game_id": "cutoff", "candidate_color": "w", "initial_fen": initial_fen, "moves": ["e2e4"],
             "final_fen": cutoff.to_fen(), "candidate_score": 1., "outcome": "*",
             "termination": "move_limit", "truncated": True}]
        (comparison / "matches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in matches))
        (root / "value_curve.json").write_text(json.dumps([
            {"game": 100, "checkpoint_sha256": "final", "metrics": values},
            {"game": 0, "checkpoint_sha256": "baseline", "metrics": dict(values, all=cls.metrics(12, 3, .7, .7, .75))}]))
        (root / "human_retention.json").write_text(json.dumps({
            "baseline": {"metrics": {"policy_nll_full_action": 6.25, "policy_sample_count": 100}},
            "candidate": {"policy_nll_full_action": 6.5, "policy_sample_count": 100}}))
        (root / "probes.jsonl").write_text(json.dumps({"game": 0, "positions": [{"passed": True}, {"passed": False}, {"passed": None}]}) + "\n")
        return root

    def test_partial_run_keeps_valid_lines_and_warns_about_incomplete_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "updates.jsonl").write_text('{"step":1,"value_loss":0.2}\n{"step":2')
            (root / "value_curve.json").write_text('{"wrong":"container"}')
            document, data = read_report(generate_report(root))
            self.assertEqual(data["updates"], [{"step": 1, "value_loss": .2}])
            self.assertEqual(data["value_curve"], [])
            self.assertEqual(data["matches"], [])
            self.assertEqual(len(data["warnings"]), 2)
            self.assertIn("No measurements yet.", document)
            self.assertNotRegex(document, r'<script[^>]+src=')

    def test_cutoffs_never_score_and_decisive_metrics_use_position_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, data = read_report(generate_report(self.fixture(temporary)))
            totals = data["match_totals"]
            self.assertEqual(totals["games"], 3)
            self.assertEqual(totals["completed_games"], 2)
            self.assertEqual(totals["candidate_wins"], 1)
            self.assertEqual(totals["actual_draws"], 1)
            self.assertEqual(totals["unresolved_cutoffs"], 1)
            self.assertEqual(totals["candidate_score_on_completed_games"], .75)
            self.assertEqual(totals["candidate_score_all_games_bounds"], [.5, 2.5 / 3])
            decisive = data["value_breakdowns"][1]["metrics"]
            self.assertEqual(decisive["positions"], 9)
            self.assertEqual(decisive["games"], 2)
            self.assertAlmostEqual(decisive["baseline"]["mse"], (3 * .8 + 6) / 9)
            self.assertAlmostEqual(decisive["candidate"]["mse"], (3 * .7 + 6 * .9) / 9)
            self.assertEqual([row["game"] for row in data["value_curve"]], [0, 100])
            self.assertEqual(data["matches"][0]["frames"][-1], data["matches"][0]["final_fen"])
            self.assertEqual(len(data["matches"][0]["frames"]), 5)

    def test_invalid_match_history_is_visible_and_does_not_prevent_other_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            path = root / "comparison/matches.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]["moves"][0] = "e2e5"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            _, data = read_report(generate_report(root))
            self.assertEqual(data["matches"][0]["frames"], [])
            self.assertEqual(len(data["matches"][2]["frames"]), 2)
            self.assertTrue(any("Cannot replay match mate" in warning for warning in data["warnings"]))
            self.assertEqual(data["match_totals"]["completed_games"], 2)

    def test_script_injection_and_nonfinite_json_are_rejected_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attack = '</script><script>alert("x")</script>&<img src=x>'
            (root / "config.json").write_text(json.dumps({"config": {"label": attack}}))
            (root / "updates.jsonl").write_text('{"step":1,"value_loss":Infinity}\n')
            document, data = read_report(generate_report(root))
            self.assertEqual(data["config"]["label"], attack)
            self.assertNotIn(attack, document)
            self.assertIn(r"\u003c/script\u003e", document)
            self.assertNotIn("innerHTML", document)
            self.assertEqual(data["updates"], [])
            self.assertEqual(len(data["warnings"]), 1)

    def test_custom_output_preserves_separate_training_report_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            original = root / "report.html"
            original.write_text("existing training inspector")
            output = root / "export/selfplay.html"
            _, data = read_report(generate_report(root, output))
            self.assertEqual(original.read_text(), "existing training inspector")
            self.assertEqual(data["training_report_href"], "./../report.html")
            self.assertFalse(list(output.parent.glob("*.tmp")))

    @unittest.skipUnless(shutil.which("node"), "Node is optional for DOM regression checks")
    def test_board_navigation_scores_and_curves_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = generate_report(self.fixture(temporary))
            subprocess.run([shutil.which("node"), "-e", DOM_HARNESS, str(path)], check=True, capture_output=True, text=True)

    @unittest.skipUnless(shutil.which("node"), "Node is optional for DOM regression checks")
    def test_empty_report_renders_without_optional_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = generate_report(temporary)
            subprocess.run([shutil.which("node"), "-e", DOM_HARNESS, str(path)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
