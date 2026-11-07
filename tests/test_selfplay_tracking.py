"""Termination accounting and inspectable self-play records."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.chess_engine import GameState

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from scr.rl_interface import NUM_ACTIONS
    from scr.self_play import play_self_game


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional training dependency")
class SelfPlayTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_terminal_reasons_and_winners(self):
        cases = [
            ("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1", "checkmate", "1-0", "w", 1.0),
            ("8/8/8/8/8/6k1/6q1/7K w - - 0 1", "checkmate", "0-1", "b", -1.0),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "stalemate", "1/2-1/2", None, 0.0),
            ("4k3/8/8/8/8/8/8/R3K3 w - - 100 1", "fifty_move_rule", "1/2-1/2", None, 0.0),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "insufficient_material", "1/2-1/2", None, 0.0),
        ]
        for fen, reason, outcome, winner, result in cases:
            with self.subTest(reason=reason, outcome=outcome):
                data, actual, info = play_self_game(GameState.from_fen(fen), None,
                                                    max_moves=0, return_info=True, record_trace=True)
                self.assertEqual(data, [])
                self.assertEqual(actual, result)
                self.assertEqual(info["termination"], reason)
                self.assertEqual(info["outcome"], outcome)
                self.assertEqual(info["winner"], winner)
                self.assertFalse(info["truncated"])
                self.assertEqual(info["plies"], 0)
                self.assertEqual(info["initial_fen"], fen)
                self.assertEqual(info["final_fen"], fen)
                self.assertEqual(info["trace"], [])
                json.dumps(info, allow_nan=False)

    def test_repetition_has_its_own_termination_reason(self):
        state = GameState()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            state.push_uci(move)
        _, result, info = play_self_game(state, None, return_info=True)
        self.assertEqual(result, 0)
        self.assertEqual(info["termination"], "threefold_repetition")
        self.assertEqual(info["outcome"], "1/2-1/2")
        self.assertFalse(info["truncated"])
        self.assertEqual(info["moves"], [])

    def test_move_limit_is_unfinished_but_retains_existing_zero_label(self):
        state = GameState()
        before = deepcopy(state.__dict__)
        data, result, info = play_self_game(state, None, num_mcts_sims=2, max_moves=2,
                                           temperature=0, add_root_dirichlet=False,
                                           return_info=True)
        self.assertEqual(state.__dict__, before)
        self.assertEqual(len(data), 2)
        self.assertEqual(result, 0)
        self.assertEqual(info["result"], 0)
        self.assertEqual(info["termination"], "move_limit")
        self.assertEqual(info["outcome"], "*")
        self.assertTrue(info["truncated"])
        self.assertIsNone(info["winner"])
        self.assertEqual(info["plies"], 2)
        self.assertEqual(len(info["moves"]), 2)
        replay = GameState.from_fen(info["initial_fen"])
        for move in info["moves"]:
            replay.push_uci(move)
        self.assertEqual(replay.to_fen(), info["final_fen"])
        self.assertEqual(info["trace"], [])
        self.assertGreaterEqual(info["duration_seconds"], 0)

    def test_terminal_final_allowed_ply_takes_precedence_over_cutoff(self):
        mate = GameState.from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        data, result, info = play_self_game(mate, None, num_mcts_sims=60, max_moves=1,
                                           temperature=0, add_root_dirichlet=False,
                                           return_info=True)
        self.assertEqual(len(data), 1)
        self.assertEqual(result, 1)
        self.assertEqual(info["termination"], "checkmate")
        self.assertFalse(info["truncated"])
        fifty = GameState.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 1")
        _, result, info = play_self_game(fifty, None, num_mcts_sims=0, max_moves=1,
                                       temperature=0, add_root_dirichlet=False,
                                       return_info=True)
        self.assertEqual(result, 0)
        self.assertEqual(info["termination"], "fifty_move_rule")
        self.assertFalse(info["truncated"])

    def test_zero_move_budget_and_legacy_return_shape(self):
        state = GameState()
        old = play_self_game(state, None, max_moves=0)
        self.assertEqual(old, ([], 0.0))
        _, _, info = play_self_game(state, None, max_moves=0, return_info=True)
        self.assertEqual(info["termination"], "move_limit")
        self.assertEqual(info["moves"], [])
        self.assertEqual(info["initial_fen"], info["final_fen"])

    def test_trace_uses_clean_priors_and_replayable_positions(self):
        class UniformModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, states):
                batch = len(states)
                return (torch.zeros(batch, NUM_ACTIONS, device=states.device),
                        torch.full((batch, 1), .25, device=states.device))

        random.seed(17)
        state = GameState()
        before = deepcopy(state.__dict__)
        data, _, info = play_self_game(state, UniformModel(), num_mcts_sims=0, max_moves=2,
                                       temperature=0, add_root_dirichlet=True, dir_epsilon=1,
                                       return_info=True, record_trace=True)
        self.assertEqual(state.__dict__, before)
        self.assertEqual(len(info["trace"]), 2)
        replay = GameState()
        for index, trace in enumerate(info["trace"]):
            self.assertEqual(trace["ply"], index + 1)
            self.assertEqual(trace["fen"], replay.to_fen())
            self.assertEqual(trace["player"], replay.player)
            legal = {move.uci() for move in replay.get_all_legal_moves(replay.player)}
            self.assertEqual(trace["legal_moves"], len(legal))
            self.assertIn(trace["move"], legal)
            self.assertIn(trace["policy_top_move"], legal)
            self.assertIn(trace["search_top_move"], legal)
            self.assertAlmostEqual(trace["value_prediction"], .25)
            self.assertAlmostEqual(trace["selected_policy_probability"], 1 / len(legal))
            self.assertAlmostEqual(trace["policy_entropy"], math.log(len(legal)))
            self.assertLessEqual(trace["search_entropy"], trace["policy_entropy"])
            self.assertGreater(trace["policy_search_kl"], 0)
            self.assertEqual(trace["sims"], 0)
            self.assertGreaterEqual(trace["search_seconds"], 0)
            self.assertEqual(len(trace["top_moves"]), 5)
            for candidate in trace["top_moves"]:
                self.assertIn(candidate["move"], legal)
                self.assertAlmostEqual(candidate["prior"], 1 / len(legal))
                self.assertEqual(candidate["visits"], 0)
                self.assertTrue(math.isfinite(candidate["q"]))
            self.assertAlmostEqual(data[index][1].sum().item(), 1, places=6)
            replay.push_uci(trace["move"])
        self.assertEqual(replay.to_fen(), info["final_fen"])
        json.dumps(info, allow_nan=False)

    def test_baseline_trace_has_no_neural_prediction_and_counts_new_sims(self):
        _, _, info = play_self_game(GameState(), None, num_mcts_sims=5, max_moves=2,
                                    add_root_dirichlet=False, return_info=True, record_trace=True)
        for trace in info["trace"]:
            self.assertIsNone(trace["value_prediction"])
            self.assertEqual(trace["sims"], 5)
            self.assertGreater(trace["selected_search_probability"], 0)
            self.assertGreater(trace["selected_policy_probability"], 0)
        json.dumps(info, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
