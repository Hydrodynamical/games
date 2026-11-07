"""Independent rule regressions and published chess perft reference counts."""
import unittest
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.chess_engine import GameState


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def perft(state, depth):
    if depth == 0:
        return 1
    moves = state.get_all_legal_moves(state.player)
    if depth == 1:
        return len(moves)
    count = 0
    for move in moves:
        state.make_move(move)
        count += perft(state, depth - 1)
        state.undo_last_move()
    return count


class EngineTests(unittest.TestCase):
    def test_perft_reference_positions(self):
        positions = [
            (START_FEN, [20, 400, 8902]),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
             [48, 2039, 97862]),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812]),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
             [6, 264, 9467]),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
             [44, 1486, 62379]),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
             [46, 2079, 89890]),
        ]
        for fen, counts in positions:
            state = GameState.from_fen(fen)
            original = deepcopy(state.__dict__)
            # Move generation may initialize the otherwise empty undo stack.
            state.undo_stack = []
            original["undo_stack"] = []
            for depth, expected in enumerate(counts, 1):
                with self.subTest(fen=fen, depth=depth):
                    self.assertEqual(perft(state, depth), expected)
                    self.assertEqual(state.__dict__, original)

    def test_fen_roundtrip_and_counters(self):
        state = GameState()
        self.assertEqual(state.to_fen(), START_FEN)
        state.push_uci("e2e4")
        self.assertEqual(state.to_fen(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        state.push_uci("g8f6")
        self.assertEqual(state.halfmove_clock, 1)
        self.assertEqual(state.turn, 2)
        self.assertEqual(GameState.from_fen(state.to_fen()).to_fen(), state.to_fen())
        state.undo_last_move()
        state.undo_last_move()
        self.assertEqual(state.to_fen(), START_FEN)
        loaded = GameState.from_fen("4k3/8/8/8/8/8/8/R3K3 b - - 17 42")
        copied = loaded.fast_copy_for_search()
        copied.push_uci("e8f8")
        self.assertEqual(copied.to_fen().split()[-2:], ["18", "43"])
        copied.undo_last_move()
        self.assertEqual(copied.to_fen(), loaded.to_fen())

    def test_bad_fen_is_rejected(self):
        for fen in ("", "8/8/8/8/8/8/8/8 w - - 0 1",
                    "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
                    "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
                    "4k3/8/8/8/8/8/8/4K3 w - e6 0 1",
                    "4k3/8/8/8/8/8/8/4K3 w - - -1 1",
                    "4k3/8/8/8/8/8/8/4K3 w - - 0 0"):
            with self.subTest(fen=fen), self.assertRaises(ValueError):
                GameState.from_fen(fen)

    def test_uci_rejects_illegal_moves_without_mutation(self):
        state = GameState()
        for move in ("e2e5", "e7e5", "e2e4q", "0000", "e9e4", "e2", "e2e4nonsense"):
            with self.subTest(move=move), self.assertRaises(ValueError):
                state.push_uci(move)
            self.assertEqual(state.to_fen(), START_FEN)
        self.assertEqual(state.parse_uci("e2e4").uci(), "e2e4")

    def test_en_passant_capture_undo_and_expiration(self):
        state = GameState.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 12")
        before = state.to_fen()
        move = state.parse_uci("e5d6")
        self.assertTrue(move.is_en_passant)
        self.assertEqual(state.move_from_pair(((3, 4), (2, 3))), move)
        state.make_move(move)
        self.assertEqual(state.board[3][3], "--")
        self.assertEqual(state.board[2][3], "wP")
        self.assertEqual(state.halfmove_clock, 0)
        state.undo_last_move()
        self.assertEqual(state.to_fen(), before)
        state.push_uci("e1f1")
        self.assertIsNone(state.en_passant_target)

    def test_en_passant_cannot_expose_king(self):
        state = GameState.from_fen("k7/8/8/4KPpr/8/8/8/8 w - g6 0 1")
        self.assertNotIn("f5g6", {m.uci() for m in state.get_all_legal_moves("w")})
        no_ep = GameState.from_fen(state.to_fen().replace("g6", "-"))
        self.assertEqual(state.position_key(), no_ep.position_key())

    def test_repetition_ignores_unavailable_ep_but_keeps_legal_ep(self):
        state = GameState()
        state.push_uci("e2e4")
        no_ep = GameState.from_fen(state.to_fen().replace("e3", "-"))
        self.assertEqual(state.position_key(), no_ep.position_key())
        state = GameState.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        no_ep = GameState.from_fen(state.to_fen().replace("d6", "-"))
        self.assertNotEqual(state.position_key(), no_ep.position_key())

    def test_all_promotion_choices_and_capture_undo(self):
        for color, fen, origin, destination, capture in (
            ("w", "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7", "a8", "b8"),
            ("b", "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1", "a2", "a1", "b1"),
        ):
            state = GameState.from_fen(fen)
            moves = {m.uci() for m in state.get_all_legal_moves(color)}
            for target in (destination, capture):
                for promotion in "qrbn":
                    uci = origin + target + promotion
                    self.assertIn(uci, moves)
                    move = state.push_uci(uci)
                    self.assertEqual(state.board[move.end[0]][move.end[1]], color + promotion.upper())
                    state.undo_last_move()
                    self.assertEqual(state.to_fen(), fen)

    def test_castling_moves_rook_and_undo_restores_rights(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        state = GameState.from_fen(fen)
        for uci, rook_col in (("e1g1", 5), ("e1c1", 3)):
            state.push_uci(uci)
            self.assertEqual(state.board[7][rook_col], "wR")
            self.assertFalse(state.castling_rights["wK"])
            self.assertFalse(state.castling_rights["wQ"])
            state.undo_last_move()
            self.assertEqual(state.to_fen(), fen)
        state.push_uci("a1a8")
        self.assertFalse(state.castling_rights["wQ"])
        self.assertFalse(state.castling_rights["bQ"])
        state.undo_last_move()
        self.assertEqual(state.to_fen(), fen)

    def test_castling_through_check_is_forbidden(self):
        state = GameState.from_fen("4kr2/8/8/8/8/8/8/4K2R w K - 0 1")
        self.assertNotIn("e1g1", {m.uci() for m in state.get_all_legal_moves("w")})

    def test_checkmate_stalemate_and_fifty_move_precedence(self):
        state = GameState()
        for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
            state.push_uci(move)
        self.assertTrue(state.is_checkmate())
        self.assertFalse(state.is_stalemate())
        state.halfmove_clock = 100
        self.assertFalse(state.is_draw())
        stale = GameState.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(stale.is_stalemate())
        self.assertFalse(stale.is_checkmate())
        fifty = GameState.from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 1")
        self.assertEqual(fifty.draw_reason(), "fifty_move_rule")

    def test_insufficient_material(self):
        for fen in ("4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                    "4k3/8/8/8/8/8/8/3BK3 w - - 0 1",
                    "4k3/8/8/8/8/8/8/3NK3 w - - 0 1",
                    "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1"):
            with self.subTest(fen=fen):
                self.assertEqual(GameState.from_fen(fen).draw_reason(), "insufficient_material")
        for fen in ("4k3/8/8/8/8/8/8/2NNK3 w - - 0 1",
                    "4k3/8/8/8/8/8/8/2BNK3 w - - 0 1",
                    "4k3/8/8/8/8/8/8/3RK3 w - - 0 1"):
            self.assertFalse(GameState.from_fen(fen).is_insufficient_material())

    def test_search_copy_preserves_repetition_and_is_independent(self):
        state = GameState()
        sequence = ("g1f3", "g8f6", "f3g1", "f6g8")
        for uci in sequence:
            state.push_uci(uci)
        clone = state.fast_copy_for_search()
        self.assertEqual(clone.position_counts, state.position_counts)
        self.assertEqual(clone._pos_stack, state._pos_stack)
        for uci in sequence:
            clone.push_uci(uci)
        self.assertTrue(clone.is_threefold_repetition())
        self.assertFalse(state.is_threefold_repetition())
        self.assertEqual(clone.draw_reason(), "threefold_repetition")
        for _ in sequence:
            clone.undo_last_move()
        self.assertEqual(clone.position_counts, state.position_counts)
        self.assertEqual(clone.to_fen(), state.to_fen())
        clone.board[0][0] = "--"
        self.assertEqual(state.board[0][0], "bR")

    def test_generating_other_side_moves_preserves_state(self):
        state = GameState()
        self.assertEqual(len(state.get_all_legal_moves("b")), 20)
        self.assertEqual(state.to_fen(), START_FEN)
        self.assertEqual(len(state._pos_stack), 1)
        self.assertEqual(sum(state.position_counts.values()), 1)
        self.assertEqual(len(state.get_legal_moves((0, 1), respect_turn_order=False)), 2)


if __name__ == "__main__":
    unittest.main()
