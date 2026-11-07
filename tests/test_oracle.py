"""Optional independent rules regression against python-chess (dev only)."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.chess_engine import GameState

try:
    import chess as reference
    if not hasattr(reference, "Board"):
        reference = None
except ImportError:
    reference = None


@unittest.skipIf(reference is None, "install requirements/dev.txt for the independent rules oracle")
class ReferenceRulesTests(unittest.TestCase):
    def test_seeded_games_match_legal_moves_state_and_undo(self):
        rng = random.Random(1307)
        for game in range(12):
            actual, expected = GameState(), reference.Board()
            for ply in range(100):
                with self.subTest(game=game, ply=ply, fen=expected.fen()):
                    moves = {move.uci(): move for move in actual.get_all_legal_moves(actual.player, as_moves=True)}
                    self.assertEqual(set(moves), {move.uci() for move in expected.legal_moves})
                    self.assertEqual(actual.to_fen(), expected.fen(en_passant="fen"))
                    self.assertEqual(actual.in_check(), expected.is_check())
                    self.assertEqual(actual.is_insufficient_material(), expected.is_insufficient_material())
                    if not moves:
                        break
                    chosen = rng.choice(sorted(moves))
                    original = actual.to_fen()
                    actual.make_move(moves[chosen])
                    actual.undo_last_move()
                    self.assertEqual(actual.to_fen(), original)
                    actual.make_move(moves[chosen])
                    expected.push_uci(chosen)

    def test_python_chess_can_drive_uci_engine(self):
        import chess.engine
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "chess" / "cli.py"), "uci"]
        with chess.engine.SimpleEngine.popen_uci(command, timeout=5) as engine:
            board = chess.Board()
            for _ in range(8):
                result = engine.play(board, chess.engine.Limit(nodes=20))
                self.assertIn(result.move, board.legal_moves)
                board.push(result.move)


if __name__ == "__main__":
    unittest.main()
