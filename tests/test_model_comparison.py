"""Two-model diagnostics preserve populations, perspectives, and cutoff status."""
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
    from scr import evaluate_models as evaluation
    from scr.chess_engine import GameState
    from scr.rl_interface import NUM_ACTIONS


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional training dependency")
class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def model(value=0., black_win=False):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.seen = []

            def forward(self, states):
                self.seen.extend(states.detach().cpu().clone())
                values = 1 - 2 * states[:, 12, 0, 0] if black_win else torch.full((len(states),), value)
                return torch.zeros((len(states), NUM_ACTIONS)) + self.anchor, values.reshape(-1, 1)

        return TinyModel().eval()

    @staticmethod
    def game(kind="mate", game_id=None):
        if kind == "mate":
            moves, result, termination, truncated = ["f2f3", "e7e5", "g2g4", "d8h4"], "0-1", "checkmate", False
        elif kind == "draw":
            moves, result, termination, truncated = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2, "1/2-1/2", "threefold_repetition", False
        else:
            moves, result, termination, truncated = ["e2e4", "e7e5"], "*", "move_limit", True
        state = GameState()
        initial_fen = state.to_fen()
        for move in moves:
            state.push_uci(move)
        return {"game_id": game_id or kind, "initial_fen": initial_fen, "moves": moves,
                "result": result, "termination": termination, "truncated": truncated, "final_fen": state.to_fen()}

    @staticmethod
    def benchmark(directory, games):
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        path = directory / "games.jsonl"
        path.write_text("".join(json.dumps(game) + "\n" for game in games))
        (directory / "manifest.json").write_text(json.dumps({"status": "completed", "checkpoint_sha256": "frozen-model",
                                                            "games_sha256": evaluation.sha256_file(path)}))
        return directory

    def test_values_use_mover_perspective_identical_batches_and_sample_weighting(self):
        baseline, candidate = self.model(), self.model(black_win=True)
        rows = []
        summary, games = evaluation.score_value_games(baseline, candidate, [self.game(), self.game("draw"), self.game("cutoff")],
                                                      batch_size=3, bootstrap_samples=40, seed=7, position_sink=rows.append)
        self.assertEqual(summary["all"]["positions"], 12)
        self.assertEqual([row["target_value"] for row in rows[:4]], [-1., 1., -1., 1.])
        self.assertTrue(all(row["target_value"] == 0 for row in rows[4:]))
        self.assertEqual(len(baseline.seen), len(candidate.seen))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline.seen, candidate.seen)))
        self.assertAlmostEqual(float(baseline.seen[8][19, 0, 0]), 2 / 3, places=6)
        self.assertAlmostEqual(summary["all"]["baseline"]["mse"], 4 / 12)
        self.assertAlmostEqual(summary["all"]["candidate"]["mse"], 8 / 12)
        self.assertAlmostEqual(summary["all"]["candidate_minus_baseline_mse"], 1 / 3)
        self.assertEqual(summary["equal_game_mean_differences"]["mse"], 0.)
        self.assertEqual(summary["by_target_outcome"]["loss"]["candidate"]["mse"], 0.)
        self.assertEqual(summary["by_target_outcome"]["win"]["candidate"]["mse"], 0.)
        self.assertEqual(summary["by_target_outcome"]["draw"]["candidate"]["mse"], 1.)
        self.assertEqual(summary["by_game_outcome"]["decisive"]["candidate"]["mse"], 0.)
        self.assertEqual(summary["by_game_outcome"]["draw"]["positions"], 8)
        self.assertEqual(summary["all"]["candidate"]["mean_prediction"], 0.)
        self.assertEqual(summary["all"]["candidate"]["mean_absolute_prediction"], 1.)
        self.assertEqual(summary["by_phase"]["later"]["positions"], 0)
        self.assertEqual(summary["excluded_unknown_outcome_games"], 1)
        self.assertEqual(summary["excluded_unknown_outcome_positions"], 2)
        self.assertEqual(games[-1]["positions_scored"], 0)
        self.assertIsNone(games[-1]["metrics"]["candidate"]["mse"])
        self.assertEqual(summary["bootstrap_game_level"], evaluation.bootstrap_games(games, 40, 7))
        self.assertEqual(summary["bootstrap_game_level"]["candidate_minus_baseline_mse"], [-1., 1.])

    def test_unknown_results_and_move_limit_labels_never_enter_value_metrics(self):
        games = [self.game("cutoff", "flag"), self.game("cutoff", "result"), self.game("cutoff", "reason")]
        games[0]["result"] = "1/2-1/2"  # Even an old training draw label is not a known result.
        games[1].update(truncated=False, termination="unknown")
        games[2].update(truncated=False, result="1/2-1/2")
        summary, _ = evaluation.score_value_games(self.model(), self.model(), games)
        self.assertEqual(summary["all"]["positions"], 0)
        self.assertIsNone(summary["all"]["candidate"]["mse"])
        self.assertEqual(summary["excluded_unknown_outcome_games"], 3)
        self.assertIsNone(summary["bootstrap_game_level"]["candidate_minus_baseline_mse"])

    def test_replay_rejects_false_outcomes_illegal_cutoffs_and_moves_after_terminal(self):
        corrupt = self.game()
        corrupt["result"] = "1-0"
        with self.assertRaisesRegex(ValueError, "outcome/termination"):
            evaluation.score_value_games(self.model(), self.model(), [corrupt])
        corrupt = self.game("cutoff")
        corrupt["moves"][0] = "e2e5"
        with self.assertRaises(ValueError):
            evaluation.score_value_games(self.model(), self.model(), [corrupt])
        corrupt = self.game("draw")
        corrupt["moves"].append("e2e4")
        with self.assertRaisesRegex(ValueError, "after terminal"):
            evaluation.score_value_games(self.model(), self.model(), [corrupt])
        corrupt = self.game()
        corrupt.update(truncated=True, result="*", termination="move_limit")
        with self.assertRaisesRegex(ValueError, "terminal final board"):
            evaluation.score_value_games(self.model(), self.model(), [corrupt])

    def test_benchmark_integrity_and_reusable_comparison_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.benchmark(directory, [self.game(), self.game("cutoff")])
            value = evaluation.compare_values(self.model(), self.model(black_win=True), root)
            self.assertEqual(value["all"]["candidate"]["mse"], 0.)
            self.assertEqual(value["benchmark_generator_checkpoint_sha256"], "frozen-model")
            self.assertEqual(value["benchmark_sha256"]["games"], evaluation.sha256_file(root / "games.jsonl"))
            with (root / "games.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                evaluation.load_benchmark(root)
            (root / "manifest.json").write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError, "not complete"):
                evaluation.load_benchmark(root)

    def test_canonical_benchmark_metadata_cannot_misstate_cutoff_population(self):
        with tempfile.TemporaryDirectory() as directory:
            game = self.game("cutoff")
            game["result"] = "1/2-1/2"
            root = self.benchmark(directory, [game])
            path = root / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(kind="frozen_selfplay_value_benchmark", counts={"games": 1})
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "cutoff/result metadata"):
                evaluation.load_benchmark(root)
            root = self.benchmark(directory, [self.game()])
            manifest = json.loads(path.read_text())
            manifest.update(kind="frozen_selfplay_value_benchmark", counts={"games": 2})
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "game count"):
                evaluation.load_benchmark(root)

    def test_openings_are_reproducible_legal_distinct_and_retain_history(self):
        openings = evaluation.build_openings(20, seed=17)
        self.assertEqual(openings, evaluation.build_openings(20, seed=17))
        self.assertNotEqual(openings[6:], evaluation.build_openings(20, seed=18)[6:])
        self.assertEqual(len({tuple(row["moves"]) for row in openings}), 20)
        for opening in openings:
            state = GameState()
            for move in opening["moves"]:
                state.push_uci(move)
            self.assertEqual(state.to_fen(), opening["fen_after_opening"])
            self.assertIsNone(evaluation.terminal_details(state))
        with self.assertRaises(ValueError):
            evaluation.build_openings(21)

    def test_search_uses_equal_budget_no_noise_fresh_root_and_deterministic_choice(self):
        state = GameState()
        move = state.parse_uci("e2e4")
        root = SimpleNamespace(edges={move: SimpleNamespace(q=.1, prior=1.)})
        with patch.object(evaluation, "mcts_search", return_value=({move: 1.}, root)) as search:
            chosen, _ = evaluation.search_decision(self.model(), state, 64)
        self.assertEqual(chosen, move)
        kwargs = search.call_args.kwargs
        self.assertEqual(kwargs["num_sims"], 64)
        self.assertEqual(kwargs["c_puct"], 1.5)
        self.assertEqual(kwargs["batch_size"], 16)
        self.assertFalse(kwargs["add_root_dirichlet"])
        self.assertIsNone(kwargs["root"])

    def test_last_allowed_ply_checkmate_and_color_assignment(self):
        baseline, candidate = self.model(), self.model()
        opening = {"pair": 1, "name": "mate", "moves": ["f2f3", "e7e5", "g2g4"]}
        calls = []

        def mate(model, state, sims):
            calls.append((model, sims))
            return state.parse_uci("d8h4"), .1

        with patch.object(evaluation, "search_decision", mate):
            black = evaluation.play_match(baseline, candidate, opening, "b", 64, 1)
            white = evaluation.play_match(baseline, candidate, opening, "w", 64, 1)
        self.assertEqual(calls, [(candidate, 64), (baseline, 64)])
        self.assertEqual(black["candidate_score"], 1.)
        self.assertEqual(white["candidate_score"], 0.)
        self.assertEqual(black["moves"], opening["moves"] + ["d8h4"])
        self.assertEqual(black["outcome"], "0-1")
        self.assertFalse(black["truncated"])
        state = GameState.from_fen(black["initial_fen"])
        for move in black["moves"]:
            state.push_uci(move)
        self.assertEqual(state.to_fen(), black["final_fen"])

    def test_cutoff_score_bounds_do_not_treat_unresolved_as_draws(self):
        game = evaluation.play_match(self.model(), self.model(), evaluation.build_openings(1)[0], "w", sims=2, max_plies=1)
        self.assertEqual(game["outcome"], "*")
        self.assertIsNone(game["candidate_score"])
        summary = evaluation.match_summary([game, {"candidate_score": 1., "termination": "checkmate"},
                                            {"candidate_score": .5, "termination": "stalemate"}])
        self.assertEqual(summary["completed_games"], 2)
        self.assertEqual(summary["actual_draws"], 1)
        self.assertEqual(summary["candidate_score_on_completed_games"], .75)
        self.assertEqual(summary["candidate_score_all_games_bounds"], [.5, 5 / 6])
        self.assertIsNone(evaluation.match_summary([game])["candidate_score_on_completed_games"])
        self.assertEqual(evaluation.match_summary([game])["candidate_score_all_games_bounds"], [0., 1.])

    def test_run_pairs_colors_records_hashes_and_refuses_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = self.benchmark(root / "benchmark", [self.game()])
            baseline, candidate = root / "baseline.pt", root / "candidate.pt"
            baseline.write_bytes(b"baseline fixture")
            candidate.write_bytes(b"candidate fixture")
            calls = []

            def match(_baseline, _candidate, opening, color, sims, plies):
                calls.append((opening, color, sims, plies))
                return {"candidate_score": None, "termination": "move_limit", "outcome": "*", "truncated": True}

            with patch.object(evaluation, "load_model", side_effect=[self.model(), self.model(black_win=True)]), \
                 patch.object(evaluation, "play_match", match):
                result = evaluation.compare_models(baseline, candidate, benchmark, root / "comparison",
                                                   match_pairs=2, sims=64, max_plies=400, threads=1, bootstrap_samples=10)
            self.assertEqual([call[1] for call in calls], ["w", "b", "w", "b"])
            self.assertEqual(calls[0][0], calls[1][0])
            self.assertEqual(calls[2][0], calls[3][0])
            self.assertTrue(all(call[2:] == (64, 400) for call in calls))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["config"]["input_sha256"]["candidate_checkpoint"], evaluation.sha256_file(candidate))
            self.assertEqual(result["values"]["all"]["positions"], 4)
            self.assertEqual(result["matches"]["unresolved_cutoffs"], 4)
            self.assertEqual(len((root / "comparison" / "value_positions.jsonl").read_text().splitlines()), 4)
            self.assertEqual(result, json.loads((root / "comparison" / "summary.json").read_text()))
            with self.assertRaises(FileExistsError):
                evaluation.compare_models(baseline, candidate, benchmark, root / "comparison")

    def test_inputs_changing_during_comparison_mark_run_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = self.benchmark(root / "benchmark", [self.game()])
            baseline, candidate = root / "baseline.pt", root / "candidate.pt"
            baseline.write_bytes(b"baseline")
            candidate.write_bytes(b"candidate")

            def match(*args):
                candidate.write_bytes(b"changed candidate")
                return {"candidate_score": None, "termination": "move_limit", "outcome": "*"}

            with patch.object(evaluation, "load_model", return_value=self.model()), patch.object(evaluation, "play_match", match):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    evaluation.compare_models(baseline, candidate, benchmark, root / "comparison", match_pairs=1, threads=1)
            summary = json.loads((root / "comparison" / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")


if __name__ == "__main__":
    unittest.main()
