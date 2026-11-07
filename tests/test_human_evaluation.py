"""A human-policy comparison keeps history, identities, and score denominators."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from scr import evaluate_human as evaluation
    from scr.chess_engine import GameState
    from scr.rl_interface import NUM_ACTIONS, move_to_action


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional training dependency")
class HumanEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def record(game_id="g1", target_color="w", result="1-0", split="test", moves=None):
        return {"game_id": game_id, "game_index": 1, "split": split,
                "initial_fen": GameState().to_fen(), "target_color": target_color,
                "result": result, "end_time": 100,
                "moves": moves or ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]}

    @staticmethod
    def write_games(directory, records):
        path = Path(directory) / "games.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        (Path(directory) / "manifest.json").write_text(json.dumps({
            "status": "completed", "games_sha256": evaluation.sha256_file(path)}))

    @staticmethod
    def model(preferred="e2e4"):
        action = move_to_action(GameState().parse_uci(preferred))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, states):
                logits = torch.full((len(states), NUM_ACTIONS), -8.0, device=states.device)
                logits[:, action] = 8.0
                return logits + self.anchor, torch.full((len(states), 1), .25, device=states.device)

        return TinyModel().eval()

    def test_sampling_is_reproducible_filters_split_and_only_target_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [self.record(str(i), target_color="b") for i in range(10)]
            records.append(self.record("must-not-appear", split="train"))
            self.write_games(directory, records)
            first, stats = evaluation.sample_positions(directory, max_positions=7, seed=9)
            second, _ = evaluation.sample_positions(directory, max_positions=7, seed=9)
            self.assertEqual([(r["game_id"], r["ply"]) for r in first],
                             [(r["game_id"], r["ply"]) for r in second])
            self.assertEqual(stats["eligible_positions"], 30)
            self.assertEqual(stats["games"], 10)
            self.assertEqual(len(first), 7)
            self.assertTrue(all(row["state"].player == "b" and row["ply"] % 2 == 0 for row in first))
            self.assertTrue(all(row["target_value"] == -1 for row in first))
            self.assertFalse(any(row["game_id"] == "must-not-appear" for row in first))

    def test_repetition_history_is_preserved_and_terminal_positions_are_counted(self):
        moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2 + ["e2e4"]
        with tempfile.TemporaryDirectory() as directory:
            self.write_games(directory, [self.record(moves=moves, result="1/2-1/2")])
            positions, stats = evaluation.sample_positions(directory, max_positions=20)
            second_start = next(row["state"] for row in positions if row["ply"] == 5)
            self.assertEqual(second_start.position_counts[second_start.position_key()], 2)
            self.assertEqual(stats["engine_terminal_exclusions"], {"threefold_repetition": 1})
            self.assertEqual([row["ply"] for row in positions], [1, 3, 5, 7])
            self.assertTrue(all(row["target_value"] == 0 for row in positions))

    def test_modified_or_partial_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_games(directory, [self.record()])
            path = Path(directory) / "games.jsonl"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                evaluation.sample_positions(directory)
            (Path(directory) / "manifest.json").write_text('{"status":"preparing"}')
            with self.assertRaisesRegex(ValueError, "not complete"):
                evaluation.sample_positions(directory)

    def test_shared_position_raw_action_nll_and_noiseless_fresh_search(self):
        state = GameState()
        human = state.parse_uci("e2e4")
        alternative = state.parse_uci("d2d4")

        def fake_search(position, model, **kwargs):
            self.assertEqual(position.to_fen(), state.to_fen())
            self.assertEqual(kwargs["num_sims"], 64)
            self.assertFalse(kwargs["add_root_dirichlet"])
            self.assertIsNone(kwargs["root"])
            distribution = {human: .25, alternative: .75}
            root = SimpleNamespace(edges={move: SimpleNamespace(q=0., prior=p) for move, p in distribution.items()})
            return distribution, root

        example = {"state": state, "game_id": "one", "ply": 1,
                   "human_move": "e2e4", "target_value": -1.0}
        with patch.object(evaluation, "mcts_search", fake_search):
            row = evaluation.evaluate_position(self.model(), example)
        self.assertEqual(row["raw_move"], "e2e4")
        self.assertEqual(row["search_move"], "d2d4")
        self.assertTrue(row["raw_match"])
        self.assertFalse(row["search_match"])
        self.assertTrue(row["move_changed"])
        self.assertLess(row["human_move_nll"], .001)
        self.assertEqual(row["human_move_search_probability"], .25)
        self.assertAlmostEqual(row["value_squared_error"], 1.5625)
        self.assertEqual(row["phase"], "opening")
        self.assertEqual(state.to_fen(), GameState().to_fen())
        summary = evaluation.position_summary([row])
        self.assertEqual(summary["all"]["raw_top1"], 1)
        self.assertEqual(summary["all"]["search_top1"], 0)
        self.assertEqual(summary["later"]["positions"], 0)

    def test_cutoffs_are_unresolved_not_draws_or_score_denominators(self):
        model = self.model()
        game = evaluation.play_match(model, "test", ("e2e4", "e7e5"), "b", sims=2, max_plies=1)
        self.assertEqual(game["outcome"], "*")
        self.assertEqual(game["termination"], "move_limit")
        self.assertIsNone(game["mcts_score"])
        summary = evaluation.match_summary([
            game, {"mcts_score": 1., "termination": "checkmate"},
            {"mcts_score": .5, "termination": "stalemate"}])
        self.assertEqual(summary["unresolved_cutoffs"], 1)
        self.assertEqual(summary["actual_draws"], 1)
        self.assertEqual(summary["completed_games"], 2)
        self.assertEqual(summary["mcts_score_on_completed_games"], .75)
        self.assertIsNone(evaluation.match_summary([game])["mcts_score_on_completed_games"])

    def test_checkmate_on_final_allowed_ply_is_a_completed_game(self):
        model = self.model()

        def mate_move(_model, state):
            return {"move": state.parse_uci("d8h4"), "seconds": 0.0}

        # Black has mate in one after the opening, and uses the raw policy.
        with patch.object(evaluation, "raw_decision", mate_move):
            game = evaluation.play_match(model, "fools_mate", ("f2f3", "e7e5", "g2g4"),
                                         "w", sims=2, max_plies=1)
        self.assertEqual(game["outcome"], "0-1")
        self.assertEqual(game["termination"], "checkmate")
        self.assertFalse(game["truncated"])
        self.assertEqual(game["mcts_score"], 0.)

    def test_checkpoint_comparison_pairs_colors_and_protects_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "data", root / "evaluation"
            dataset.mkdir()
            (dataset / "manifest.json").write_text('{"status":"completed"}')
            self.write_games(dataset, [self.record()])
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"test checkpoint fixture")
            calls = []

            def fake_match(model, name, opening, color, sims, max_plies):
                calls.append((name, tuple(opening), color))
                return {"mcts_score": None, "termination": "move_limit", "outcome": "*"}

            with patch.object(evaluation, "load_model", return_value=self.model()), \
                 patch.object(evaluation, "play_match", fake_match):
                summary = evaluation.evaluate_checkpoint(checkpoint, dataset, output, max_positions=2,
                                                         sims=2, threads=1, match_pairs=2, max_plies=1)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][:2], calls[1][:2])
            self.assertEqual([call[2] for call in calls], ["w", "b", "w", "b"])
            self.assertEqual(summary["matches"]["unresolved_cutoffs"], 4)
            self.assertEqual(summary["heldout"]["all"]["positions"], 2)
            self.assertEqual(len((output / "positions.jsonl").read_text().splitlines()), 2)
            self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
            with self.assertRaises(FileExistsError):
                evaluation.evaluate_checkpoint(checkpoint, dataset, output)


if __name__ == "__main__":
    unittest.main()
