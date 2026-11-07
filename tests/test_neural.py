"""Optional neural integration tests; the stdlib engine works without PyTorch."""
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.chess_engine import GameState, Move

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from scr.rl_interface import (
        NUM_ACTIONS, NUM_INPUT_PLANES, PolicyValueNet, move_to_action,
        action_to_start_end, action_to_promotion, encode_gamestate,
        legal_action_mask, policy_distribution, temperature_distribution,
        pick_move_from_dist, save_model, load_model,
    )
    from scr.self_play import play_self_game
    from scr.train import train_step
    from scr.mcts import mcts_search


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional training dependency")
class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.model = PolicyValueNet(hidden=8)

    def test_promotions_have_unique_roundtrippable_actions(self):
        actions = set()
        for color, start_row, end_row in (("w", 1, 0), ("b", 6, 7)):
            for source_file in range(8):
                for target_file in range(max(0, source_file - 1), min(8, source_file + 2)):
                    for promotion in "QNBR":
                        move = Move((start_row, source_file), (end_row, target_file), color + "P", promotion=promotion)
                        action = move_to_action(move)
                        self.assertNotIn(action, actions)
                        actions.add(action)
                        self.assertLess(action, NUM_ACTIONS)
                        self.assertEqual(action_to_start_end(action), (move.start, move.end))
                        if promotion != "Q":
                            self.assertEqual(action_to_promotion(action), promotion)
        self.assertEqual(move_to_action(Move((6, 4), (4, 4), "wP")), 52 * 64 + 36)

    def test_state_includes_rule_state(self):
        gs = GameState()
        initial = encode_gamestate(gs)
        self.assertEqual(tuple(initial.shape), (NUM_INPUT_PLANES, 8, 8))
        gs.castling_rights["wK"] = False
        gs.en_passant_target = (2, 3)
        gs.halfmove_clock = 50
        gs.position_counts[gs.position_key()] = 2
        changed = encode_gamestate(gs)
        self.assertTrue(torch.equal(initial[:13], changed[:13]))
        self.assertEqual(changed[13].sum().item(), 0)
        self.assertEqual(changed[17, 2, 3].item(), 1)
        self.assertAlmostEqual(changed[18, 0, 0].item(), .5)
        self.assertAlmostEqual(changed[19, 0, 0].item(), 2 / 3)

    def test_neural_policy_masks_and_preserves_promotions(self):
        gs = GameState.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        legal = gs.get_all_legal_moves(gs.player, as_moves=True)
        self.assertEqual(legal_action_mask(gs).sum().item(), len(legal))
        policy = policy_distribution(self.model, gs)
        self.assertEqual(set(policy), set(legal))
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=6)
        self.assertEqual(len({move_to_action(m) for m in policy if m.promotion}), 4)

    def test_temperature_is_applied_stably(self):
        scaled = temperature_distribution({"a": .8, "b": .2, "c": 0.0}, 2.0)
        self.assertAlmostEqual(scaled["a"], 2 / 3)
        self.assertEqual(scaled["c"], 0)
        sharp = temperature_distribution({"a": .8, "b": .2}, 1e-12)
        self.assertEqual(sharp, {"a": 1.0, "b": 0.0})
        self.assertEqual(pick_move_from_dist({"a": .2, "b": .8}, 0), "b")
        with self.assertRaises(ValueError):
            temperature_distribution({"a": 1}, -1)

    def test_terminal_self_play_never_creates_empty_policy_samples(self):
        mate = GameState.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1")
        data, result = play_self_game(mate, None, max_moves=4)
        self.assertEqual(data, [])
        self.assertEqual(result, 1.0)
        draw = GameState.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        self.assertEqual(play_self_game(draw, None), ([], 0.0))

    def test_self_play_can_give_check_and_deliver_mate(self):
        gs = GameState.from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        data, result = play_self_game(gs, None, num_mcts_sims=60, max_moves=1,
                                      temperature=0, add_root_dirichlet=False)
        self.assertEqual(len(data), 1)
        self.assertEqual(result, 1.0)

    def test_neural_search_and_self_play_make_trainable_targets(self):
        gs = GameState()
        initial = gs.to_fen()
        data, result = play_self_game(gs, self.model, num_mcts_sims=2,
                                      max_moves=2, temperature=0,
                                      add_root_dirichlet=False)
        self.assertEqual(gs.to_fen(), initial)
        self.assertEqual(result, 0)
        self.assertEqual(len(data), 2)
        for state, policy, player in data:
            self.assertEqual(tuple(state.shape), (NUM_INPUT_PLANES, 8, 8))
            self.assertEqual(tuple(policy.shape), (NUM_ACTIONS,))
            self.assertAlmostEqual(policy.sum().item(), 1.0)
            self.assertIn(player, ("w", "b"))
        before = self.model.trunk[0].weight.detach().clone()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        batch = [(state, policy, result if player == "w" else -result)
                 for state, policy, player in data]
        stats = train_step(self.model, optimizer, batch, "cpu")
        self.assertTrue(all(math.isfinite(v) for v in stats.values()))
        self.assertFalse(torch.equal(before, self.model.trunk[0].weight))

    def test_compatible_checkpoint_roundtrip_and_legacy_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(self.model, path)
            restored = load_model(path, "cpu")
            self.assertTrue(torch.equal(self.model.trunk[0].weight, restored.trunk[0].weight))
            torch.save({"trunk.0.weight": torch.zeros(8, 13, 3, 3),
                        "policy_head.3.weight": torch.zeros(4096, 1)}, path)
            with self.assertRaisesRegex(ValueError, "old input/action encoding"):
                load_model(path, "cpu")


if __name__ == "__main__":
    unittest.main()
