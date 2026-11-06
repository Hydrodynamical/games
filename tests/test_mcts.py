"""Search regressions covering legal play, terminal values, budgets and NN batches."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.chess_engine import GameState
from scr.mcts import MCTSDiagnostics, Node, best_move, evaluate_position, mcts_search


class MCTSTests(unittest.TestCase):
    def test_baseline_needs_no_third_party_packages(self):
        source_dir = str(Path(__file__).resolve().parents[1] / "chess")
        code = (
            f"import sys; sys.path.insert(0, {source_dir!r}); "
            "from scr.chess_engine import GameState; from scr.mcts import mcts_search; "
            "pi = mcts_search(GameState(), num_sims=2); "
            "assert len(pi) == 20; assert abs(sum(pi.values()) - 1) < 1e-10; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        )
        subprocess.run([sys.executable, "-S", "-c", code], check=True, capture_output=True)

    def test_legal_distribution_budget_and_immutability(self):
        gs = GameState()
        before = deepcopy(gs.__dict__)
        diagnostics = MCTSDiagnostics()
        pi, root = mcts_search(gs, num_sims=37, diagnostics=diagnostics, return_root=True)
        self.assertEqual(gs.__dict__, before)
        self.assertEqual(set(pi), set(gs.get_all_legal_moves(gs.player)))
        self.assertAlmostEqual(sum(pi.values()), 1.0)
        self.assertTrue(all(math.isfinite(p) and p >= 0 for p in pi.values()))
        self.assertEqual(diagnostics.sims, 37)
        self.assertEqual(sum(edge.n for edge in root.edges.values()), 37)
        self.assertIsNotNone(best_move(pi, root))

    def test_zero_simulations_use_normalized_priors(self):
        pi, root = mcts_search(GameState(), num_sims=0, return_root=True)
        self.assertEqual(len(pi), 20)
        self.assertAlmostEqual(sum(pi.values()), 1.0)
        self.assertEqual(sum(edge.n for edge in root.edges.values()), 0)

    def test_cancellation_and_time_budget(self):
        event = threading.Event()
        event.set()
        for kwargs in ({"stop_event": event}, {"time_limit": 0}):
            diagnostics = MCTSDiagnostics()
            pi = mcts_search(GameState(), num_sims=1000000, diagnostics=diagnostics, **kwargs)
            self.assertEqual(diagnostics.sims, 0)
            self.assertAlmostEqual(sum(pi.values()), 1.0)
        start = time.monotonic()
        diagnostics = MCTSDiagnostics()
        mcts_search(GameState(), num_sims=1000000, time_limit=0.015, diagnostics=diagnostics)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertLess(diagnostics.sims, 1000000)

    def test_terminal_positions_and_checkmate_precedence(self):
        for fen, expected, reason in (
            ("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1", -1.0, "checkmate"),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.0, "stalemate"),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0.0, "draw"),
            ("4k3/8/8/8/8/8/8/R3K3 w - - 100 1", 0.0, "draw"),
        ):
            with self.subTest(fen=fen):
                state = GameState.from_fen(fen)
                pi, root = mcts_search(state, return_root=True)
                self.assertEqual(pi, {})
                self.assertEqual(root.terminal_value(), expected)
                self.assertEqual(root.terminal_reason, reason)
                self.assertIsNone(best_move(pi, root))

    def test_static_value_is_from_side_to_move(self):
        white = GameState.from_fen("7k/8/8/8/8/8/8/3Q2K1 w - - 0 1")
        black = GameState.from_fen("7k/8/8/8/8/8/8/3Q2K1 b - - 0 1")
        self.assertGreater(evaluate_position(white), 0)
        self.assertAlmostEqual(evaluate_position(white), -evaluate_position(black))

    def test_finds_immediate_mate_with_correct_backprop_sign(self):
        gs = GameState.from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        diagnostics = MCTSDiagnostics()
        pi, root = mcts_search(gs, num_sims=100, return_root=True, diagnostics=diagnostics)
        move = best_move(pi, root)
        gs.make_move(move)
        self.assertTrue(gs.is_checkmate())
        self.assertEqual(root.edges[move].q, 1.0)
        self.assertGreater(diagnostics.terminal_checkmate, 0)

    def test_prefers_winning_queen_capture(self):
        gs = GameState.from_fen("4k3/8/8/8/8/8/4q3/4R1K1 w - - 0 1")
        pi, root = mcts_search(gs, num_sims=100, return_root=True)
        self.assertEqual(best_move(pi, root).uci(), "e1e2")

    def test_reuse_requires_matching_state_and_history(self):
        gs = GameState()
        _, root = mcts_search(gs, num_sims=2, return_root=True)
        _, reused = mcts_search(gs, num_sims=0, root=root, return_root=True)
        self.assertIs(reused, root)
        altered = deepcopy(gs)
        altered.halfmove_clock = 99
        _, different = mcts_search(altered, num_sims=0, root=root, return_root=True)
        self.assertIsNot(different, root)
        altered = deepcopy(gs)
        altered.position_counts[altered.position_key()] = 2
        _, different = mcts_search(altered, num_sims=0, root=root, return_root=True)
        self.assertIsNot(different, root)
        gs.push_uci("e2e4")
        _, different = mcts_search(gs, num_sims=0, root=root, return_root=True)
        self.assertIsNot(different, root)
        self.assertEqual(different.gs.to_fen(), gs.to_fen())

    def test_root_noise_is_normalized_and_can_be_removed(self):
        gs = GameState()
        original, root = mcts_search(gs, num_sims=0, return_root=True)
        noisy, root = mcts_search(gs, num_sims=0, root=root, return_root=True,
                                  add_root_dirichlet=True, dir_epsilon=1.0)
        self.assertAlmostEqual(sum(noisy.values()), 1.0)
        self.assertNotEqual(noisy, original)
        restored = mcts_search(gs, num_sims=0, root=root)
        self.assertEqual(restored, original)

    def test_invalid_search_parameters_fail_clearly(self):
        for kwargs in ({"num_sims": -1}, {"num_sims": 1.5}, {"batch_size": 0},
                       {"c_puct": float("nan")}, {"dir_alpha": 0},
                       {"dir_epsilon": 2}, {"time_limit": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mcts_search(GameState(), **kwargs)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch is not installed")
class NeuralMCTSTests(unittest.TestCase):
    @staticmethod
    def make_model(value=0.0):
        import torch
        from scr.rl_interface import NUM_ACTIONS

        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()))
                self.batches = []

            def forward(self, x):
                self.batches.append(len(x))
                return (torch.zeros(len(x), NUM_ACTIONS, device=x.device),
                        torch.full((len(x), 1), value, device=x.device))

        return ConstantModel()

    def test_batched_neural_visits_and_virtual_losses_are_finalized(self):
        model = self.make_model()
        diagnostics = MCTSDiagnostics()
        pi, root = mcts_search(GameState(), model, num_sims=37, batch_size=8,
                               return_root=True, diagnostics=diagnostics)
        self.assertEqual(diagnostics.sims, 37)
        self.assertEqual(diagnostics.expanded_nonterminal, 37)
        self.assertEqual(sum(edge.n for edge in root.edges.values()), 37)
        self.assertTrue(all(edge.w == 0 for edge in root.edges.values()))
        self.assertAlmostEqual(sum(pi.values()), 1.0)
        self.assertTrue(any(size > 1 for size in model.batches))
        self.assertTrue(all(size <= 8 for size in model.batches))

    def test_neural_leaf_value_changes_sign_for_parent(self):
        _, root = mcts_search(GameState(), self.make_model(0.5), num_sims=1, return_root=True)
        visited = [edge for edge in root.edges.values() if edge.n]
        self.assertEqual(len(visited), 1)
        self.assertEqual(visited[0].q, -0.5)

    def test_failed_neural_batch_rolls_back_virtual_visits(self):
        model = self.make_model()
        gs = GameState()
        _, root = mcts_search(gs, model, num_sims=0, return_root=True)
        original_forward = model.forward

        def malformed_forward(x):
            logits, values = original_forward(x)
            return logits, values[:-1]

        model.forward = malformed_forward
        with self.assertRaisesRegex(ValueError, "one value per position"):
            mcts_search(gs, model, num_sims=8, root=root, batch_size=8)
        self.assertTrue(all(edge.n == 0 and edge.w == 0 for edge in root.edges.values()))
        self.assertTrue(all(not node.is_expanded for node in root.children.values()))
        model.forward = original_forward
        mcts_search(gs, model, num_sims=8, root=root, batch_size=8)
        self.assertEqual(sum(edge.n for edge in root.edges.values()), 8)
        self.assertTrue(all(edge.w == 0 for edge in root.edges.values()))


if __name__ == "__main__":
    unittest.main()
