'''This module contains a class GameState, and a dataclass Move.
    GameState is responsible for storing current state of the game, determining available move states, storing game history, and checking if move is available. 
    Move is responsible for storing information about a particular move, including special moves like en passant and castling.
'''

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable

# type alias for square coordinates
Sq = Tuple[int, int]

# UI / legacy square coordinates (row, col) as a list
UISq = list[int]

@dataclass(frozen=True, slots=True)
class Move:
    """
    Immutable, hashable chess move descriptor (safe as dict keys for MCTS).
    Undo snapshots do NOT live here.
    """
    start: Sq
    end: Sq
    piece_moved: str
    piece_captured: str = "--"

    # special moves
    is_en_passant: bool = False
    ep_captured_square: Optional[Sq] = None

    is_castle: bool = False
    rook_start: Optional[Sq] = None
    rook_end: Optional[Sq] = None

    # optional (for later)
    promotion: str = ""


@dataclass(slots=True)
class UndoRecord:
    """
    Undo information for a move. Can contain dicts/lists; does not need to be hashable.
    """
    mv: Move
    prev_en_passant_target: Optional[Sq]
    prev_castling_rights: Optional[Dict[str, bool]]
    prev_halfmove_clock: int  # NEW

class GameState():
    """
    Mutable chess position and rules engine.

    This class owns the board state and implements:
      (1) pseudo-legal move generation ("available moves"): moves consistent with
          piece motion and occupancy, but not filtered for king safety.
      (2) legal move generation: pseudo-legal moves filtered by rejecting moves
          that leave the moving side's king in check.
      (3) state transitions: applying and undoing moves using `Move` objects.

    Board representation
    --------------------
    `self.board` is an 8x8 list of lists of strings. Each square is:
      - "--" for empty
      - "<color><piece>" for a piece, e.g. "wP", "bK", where color ∈ {"w","b"}
        and piece ∈ {"P","N","B","R","Q","K"}.

    Special-move state
    ------------------
    `en_passant_target` and `castling_rights` store the extra state needed for
    en passant and castling (some rules may be implemented later).

    History / undo model
    --------------------
    Moves are logged in `move_log` as `Move` objects. Each `Move` may include
    snapshots of special-move state (e.g. prior castling rights) so that
    `undo_last_move()` can restore the exact previous state.
    """
    def __init__(self: "GameState") -> None:
        """
        Initialize a new chess game state.

        Sets up the initial board position, player turn, move history,
        and state required for special moves (en passant and castling).
        """
        # piece encoding:
        # "b" = black
        # "w" = white
        # "R" = rook
        # "N" = knight
        # "B" = bishop
        # "Q" = queen
        # "K" = king
        # "P" = pawn
        # "--" = empty_square
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]]
        
        # player is either white or black, and white starts first
        self.player = "w" 

        # turn number starts at 0
        self.turn = 0

        # board history is empty list
        self.history = []

        # NEW: move history as Move objects (will replace self.history)
        self.move_log = []

        # NEW: state for special moves (used later)
        self.en_passant_target = None
        self.castling_rights = {
        "wK": True,  # White king-side castling (O-O)
        "wQ": True,  # White queen-side castling (O-O-O)
        "bK": True,  # Black king-side castling
        "bQ": True,  # Black queen-side castling
        }

        # easy access for board dimension 
        self.board_dimension = 8

        # NEW: draw bookkeeping
        self.halfmove_clock: int = 0                # plies since last pawn move or capture
        self.position_counts: Dict[tuple, int] = {} # key -> count
        self._pos_stack: list[tuple] = []           # stack of keys for undo

        # Record initial position once
        self._record_position()

    def print_board(self: "GameState", board_dimensions: int = 8) -> None:
        """
        Print the current board position, active player, and turn number in a human-readable format.
        """
        # print aesthetic divider
        print("\n" + "=" * 48)
        print("Current board position:\n")
        # print board
        for row in range(board_dimensions):
            row_string = str(self.board[row])
            print(row_string + "\n")
        
        # print current player
        print("Current player is: " + self.player)

        # print turn number
        print("Turn number = " + str(self.turn))

        # print aesthetic divider
        print("\n" + "=" * 48)
    
    def print_move_log(self : "GameState", last_n_moves: int | None = None):
        """
        Print the move history stored in move_log.

        Parameters
        ----------
        last_n_moves : int or None
            If given, print only the last N moves.
            If None, print the full history.
        """
        # print aesthetic divider
        print("\n" + "=" * 48)
        print("Move history:\n")

        if not hasattr(self, "move_log") or len(self.move_log) == 0:
            print("  (empty)")
            return

        moves = self.move_log
        if last_n_moves is not None:
            moves = moves[-last_n_moves:]

        for i, mv in enumerate(moves, start=1):
            special = []
            if getattr(mv, "is_castle", False):
                special.append("castle")
            if getattr(mv, "is_en_passant", False):
                special.append("en-passant")

            special_str = f" [{', '.join(special)}]" if special else ""
            print(
                f"  {i:>3}: {mv.piece_moved} {mv.start} -> {mv.end}"
                f"  cap={mv.piece_captured}{special_str}"
            )
        # print aesthetic divider
        print("\n" + "=" * 48)

    # ----------------------------
    # Draw detection helpers
    # ----------------------------
    def position_key(self) -> tuple:
        """
        Hashable representation of the current position for repetition detection.

        Includes:
          - board placement
          - side to move
          - castling rights
          - en passant target
        """
        board_tuple = tuple(tuple(row) for row in self.board)

        cr = getattr(self, "castling_rights", None)
        if isinstance(cr, dict):
            # stable ordering
            cr_tuple = tuple(sorted((k, bool(v)) for k, v in cr.items()))
        else:
            cr_tuple = ()

        ep = getattr(self, "en_passant_target", None)

        return (board_tuple, self.player, cr_tuple, ep)

    def _record_position(self) -> None:
        """Increment repetition count and push current key onto the stack."""
        k = self.position_key()
        self._pos_stack.append(k)
        self.position_counts[k] = self.position_counts.get(k, 0) + 1

    def _unrecord_position(self) -> None:
        """Pop current key from the stack and decrement repetition count."""
        if not getattr(self, "_pos_stack", None):
            return
        k = self._pos_stack.pop()
        cnt = self.position_counts.get(k, 0)
        if cnt <= 1:
            self.position_counts.pop(k, None)
        else:
            self.position_counts[k] = cnt - 1

    def is_threefold_repetition(self) -> bool:
        """True iff the current position has occurred at least 3 times."""
        if not getattr(self, "_pos_stack", None):
            return False
        k = self._pos_stack[-1]
        return self.position_counts.get(k, 0) >= 3

    def is_fifty_move_draw(self) -> bool:
        """
        True iff 50-move rule is reached.

        halfmove_clock counts plies (half-moves), so threshold is 100 plies.
        """
        return getattr(self, "halfmove_clock", 0) >= 100

    def draw_reason(self) -> Optional[str]:
        if self.is_threefold_repetition():
            return "threefold_repetition"
        if self.is_fifty_move_draw():
            return "fifty_move_rule"
        return None

    def is_draw(self) -> bool:
        return self.draw_reason() is not None

    def get_board_string(self: "GameState") -> str:
        """
        Return a string representation of the current board position.

        Each row of the board is rendered on its own line.
        """
        board_string = ""
        for row in self.board:
            board_string = board_string + str(row) + "\n"
        return board_string

    def switch_player(self: "GameState") -> None:
        """
        Switch the active player from white to black or vice versa.
        """
        if self.player == "w":
            self.player = "b"
        else:
            self.player = "w"

    def move_from_pair(self: "GameState", move_pair: Sq) -> Move:
        """
        Construct a Move object from a pair of start and end squares.

        Parameters
        ----------
        move_pair : ((int, int), (int, int))
            Start and end coordinates.

        Returns
        -------
        Move
            A Move object representing the action on the current board.
        """
        (r0, c0), (r1, c1) = move_pair
        piece_moved = self.board[r0][c0]
        piece_captured = self.board[r1][c1]

        # detect castling when UI provides only (start,end)
        is_castle = False
        rook_start = None
        rook_end = None

        # Castling is a king move two files horizontally from the home file (e-file).
        if piece_moved.endswith("K") and r0 == r1 and abs(c1 - c0) == 2:
            # king-side: e->g, queen-side: e->c
            if c1 == 6:
                is_castle = True
                rook_start = (r0, 7)
                rook_end = (r0, 5)
                piece_captured = "--"  # destination square must be empty for castling
            elif c1 == 2:
                is_castle = True
                rook_start = (r0, 0)
                rook_end = (r0, 3)
                piece_captured = "--"

        return Move(
            start=(r0, c0),
            end=(r1, c1),
            piece_moved=piece_moved,
            piece_captured=piece_captured,
            is_castle=is_castle,
            rook_start=rook_start,
            rook_end=rook_end,
        )

    def make_move(self: "GameState", mv: Move, *, record_legacy_history: bool = False) -> None:
        """
        Apply a Move to the board and update game state.

        Stores undo information in GameState (not inside Move),
        so Move can remain immutable/hashable for MCTS.
        """
        # --- snapshot state for undo ---
        prev_ep = getattr(self, "en_passant_target", None)
        prev_cr = getattr(self, "castling_rights", None)
        if isinstance(prev_cr, dict):
            prev_cr = prev_cr.copy()
        prev_hmc = getattr(self, "halfmove_clock", 0)  # NEW

        # initialize undo stack if needed
        if not hasattr(self, "undo_stack"):
            self.undo_stack = []
        self.undo_stack.append(
            UndoRecord(mv=mv, 
                        prev_en_passant_target=prev_ep, 
                        prev_castling_rights=prev_cr,
                        prev_halfmove_clock=prev_hmc,
            )
        )

        (r0, c0), (r1, c1) = mv.start, mv.end

        # --- 50-move bookkeeping: reset on pawn move or capture, else +1 ---
        is_pawn_move = (mv.piece_moved[1] == "P")
        is_capture = (mv.piece_captured != "--")
        if is_pawn_move or is_capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock = getattr(self, "halfmove_clock", 0) + 1

        # apply move on board (basic)
        self.board[r0][c0] = "--"
        self.board[r1][c1] = mv.piece_moved

        # --- castling: move the rook too ---
        if mv.is_castle and mv.rook_start is not None and mv.rook_end is not None:
            rr0, cc0 = mv.rook_start
            rr1, cc1 = mv.rook_end
            rook_piece = self.board[rr0][cc0]
            self.board[rr0][cc0] = "--"
            self.board[rr1][cc1] = rook_piece

        # --- promotion: auto-queen on back rank ---
        if mv.piece_moved[1] == "P":
            promo_row = 0 if mv.piece_moved[0] == "w" else 7
            if r1 == promo_row:
                self.board[r1][c1] = mv.piece_moved[0] + "Q"

        # --- en passant target (for repetition key + later EP rules) ---
        # Set only on a 2-square pawn advance, otherwise clear.
        if mv.piece_moved[1] == "P" and abs(r1 - r0) == 2:
            mid_row = (r0 + r1) // 2
            self.en_passant_target = (mid_row, c0)
        else:
            self.en_passant_target = None
        

        # --- update castling rights ---
        if hasattr(self, "castling_rights") and isinstance(self.castling_rights, dict):
            mover_color = mv.piece_moved[0]
            enemy_color = "b" if mover_color == "w" else "w"

            # king moved => loses both castling rights
            if mv.piece_moved[1] == "K":
                self.castling_rights[mover_color + "K"] = False
                self.castling_rights[mover_color + "Q"] = False

            # rook moved from its home square => loses that side
            if mv.piece_moved[1] == "R":
                if mover_color == "w":
                    if mv.start == (7, 0):
                        self.castling_rights["wQ"] = False
                    elif mv.start == (7, 7):
                        self.castling_rights["wK"] = False
                else:
                    if mv.start == (0, 0):
                        self.castling_rights["bQ"] = False
                    elif mv.start == (0, 7):
                        self.castling_rights["bK"] = False

            # rook captured on its home square => captured side loses that side
            if mv.piece_captured.endswith("R"):
                if enemy_color == "w":
                    if mv.end == (7, 0):
                        self.castling_rights["wQ"] = False
                    elif mv.end == (7, 7):
                        self.castling_rights["wK"] = False
                else:
                    if mv.end == (0, 0):
                        self.castling_rights["bQ"] = False
                    elif mv.end == (0, 7):
                        self.castling_rights["bK"] = False

        # log move
        self.move_log.append(mv)

        # legacy history (expensive) — skip during MCTS
        if record_legacy_history:
            self.history.append(deepcopy(self.board))

        # advance turn
        self.turn += 1
        self.switch_player()

        # NEW: record the resulting position (after side-to-move flips)
        self._record_position()

    def _mk_move(self: "GameState", start: Sq, end: Sq) -> Move:
        """
        Internal helper to construct a Move from start and end squares
        using the current board state.
        """
        r0, c0 = start
        r1, c1 = end
        return Move(
            start=start,
            end=end,
            piece_moved=self.board[r0][c0],
            piece_captured=self.board[r1][c1],
        )

    def undo_last_move(self: "GameState", *, record_legacy_history: bool = False) -> None:
        """
        Undo the most recently executed move.

        Restores board position, player turn, and special-move state.
        """
        if not self.move_log:
            return
        
        # NEW: unrecord current position (the one we're leaving)
        if hasattr(self, "_pos_stack"):
            self._unrecord_position()


        mv = self.move_log.pop()

        # revert turn
        self.turn -= 1
        self.switch_player()

        (r0, c0), (r1, c1) = mv.start, mv.end

        # undo board
        self.board[r0][c0] = mv.piece_moved
        self.board[r1][c1] = mv.piece_captured

        # undo castling rook move
        if mv.is_castle and mv.rook_start is not None and mv.rook_end is not None:
            rr0, cc0 = mv.rook_start
            rr1, cc1 = mv.rook_end
            rook_piece = self.board[rr1][cc1]
            self.board[rr1][cc1] = "--"
            self.board[rr0][cc0] = rook_piece

        # restore state snapshots from undo stack
        if hasattr(self, "undo_stack") and self.undo_stack:
            rec = self.undo_stack.pop()
            if hasattr(self, "en_passant_target"):
                self.en_passant_target = rec.prev_en_passant_target
            if hasattr(self, "castling_rights") and rec.prev_castling_rights is not None:
                self.castling_rights = rec.prev_castling_rights

            # NEW: restore halfmove clock
            self.halfmove_clock = rec.prev_halfmove_clock
            
            if record_legacy_history and self.history:
                self.history.pop()

    def opponent_color(self: "GameState") -> str:
        """
        Return the color of the opponent of the current player.

        Returns
        -------
        str
            "w" if the current player is black, otherwise "b".
        """
        if self.player == "b":
            return "w"
        else:
            return "b"

    def pawn_moves(self: "GameState", start: Sq, board_dimensions: int = 8, respect_turn_order: bool = True, is_attack: bool = False) -> list[Move]:
        """
        Generate pseudo-legal pawn moves from a given square.

        Parameters
        ----------
        start : (int, int)
            Pawn starting square.
        is_attack : bool
            If True, generate attack squares only (used for attack maps).

        Returns
        -------
        list[Move]
            Pseudo-legal pawn moves.
        """

        r, c = start                # unpack start square
        piece = self.board[r][c]    # get piece at start square 
        # throw error if piece is not pawn
        if piece[1] != "P" and piece != "--":
            raise ValueError("pawn_moves called on non-pawn piece")
        # if empty square, return no moves
        if piece == "--": 
            return []
        color = piece[0]    # get color of pawn
        # if turn order is respected and pawn color is not current player, return no moves
        if respect_turn_order and color != self.player:
            return []

        moves = []

        # white pawn logic
        if color == "w":
            fwd = (r - 1, c)        # forward move
            diagL = (r - 1, c - 1)  # diagonal left attack
            diagR = (r - 1, c + 1)  # diagonal right attack
            start_rank = 6
            two = (r - 2, c)        # two steps forward from start rank
            enemy = "b"

            # if we're only generating attack moves, return them
            if is_attack:
                # pawns attack diagonally forward regardless of occupancy
                if r - 1 >= 0:
                    if c - 1 >= 0: moves.append(self._mk_move((r, c), diagL))
                    if c + 1 < board_dimensions: moves.append(self._mk_move((r, c), diagR))
                return moves 

            # one step
            if r - 1 >= 0 and self.board[fwd[0]][fwd[1]] == "--":
                moves.append(self._mk_move((r, c), fwd)) 
                # two step
                if r == start_rank and self.board[two[0]][two[1]] == "--":
                    moves.append(self._mk_move((r, c), two))

            # captures
            if r - 1 >= 0 and c - 1 >= 0 and self.board[diagL[0]][diagL[1]][0] == enemy:
                moves.append(self._mk_move((r, c), diagL))
            if r - 1 >= 0 and c + 1 < board_dimensions and self.board[diagR[0]][diagR[1]][0] == enemy:
                moves.append(self._mk_move((r, c), diagR))

        # black pawn logic
        else:
            fwd = (r + 1, c)        # forward move
            diagL = (r + 1, c - 1)  # diagonal left attack
            diagR = (r + 1, c + 1)  # diagonal right attack
            start_rank = 1
            two = (r + 2, c)        # two steps forward from start rank
            enemy = "w"

            # if we're only generating attack moves, return them
            if is_attack:
                if r + 1 < board_dimensions:
                    if c - 1 >= 0: moves.append(self._mk_move((r, c), diagL))
                    if c + 1 < board_dimensions: moves.append(self._mk_move((r, c), diagR))
                return moves

            # one step
            if r + 1 < board_dimensions and self.board[fwd[0]][fwd[1]] == "--":
                moves.append(self._mk_move((r, c), fwd))
                # two step
                if r == start_rank and self.board[two[0]][two[1]] == "--":
                    moves.append(self._mk_move((r, c), two))

            # captures
            if r + 1 < board_dimensions and c - 1 >= 0 and self.board[diagL[0]][diagL[1]][0] == enemy:
                moves.append(self._mk_move((r, c), diagL))
            if r + 1 < board_dimensions and c + 1 < board_dimensions and self.board[diagR[0]][diagR[1]][0] == enemy:
                moves.append(self._mk_move((r, c), diagR))

        return moves

    def rook_moves(self: "GameState", start: Sq, board_dimensions: int = 8, respect_turn_order: bool = True, **kwargs) -> list[Move]:
        """
        Generate pseudo-legal rook moves from a given square.

        Parameters
        ----------
        start : (int, int)
            Rook starting square.
        board_dimensions : int, optional (default=8)
            Size of the board (assumed square).
        respect_turn_order : bool, optional (default=True)
            If True, return no moves unless the rook color matches `self.player`.
            If False, generate moves regardless of whose turn it is (useful for
            generating opponent move sets / attack maps).

        Returns
        -------
        list[Move]
            Pseudo-legal rook moves (captures included). Moves are not filtered
            for king safety (i.e., they may be illegal in check contexts).

        Notes
        -----
        Any extra keyword arguments are ignored (present for a uniform dispatcher
        interface).
        """
        r, c = start               # unpack start square
        piece = self.board[r][c]   # get piece at start square
        # throw error if piece is not rook
        if piece[1] != "R" and piece != "--":
            raise ValueError("rook_moves called on non-rook piece")
        # if empty square, return no moves
        if piece == "--":
            return []
        color = piece[0]   # get color of rook
        # if turn order is respected and rook color is not current player, return no moves
        if respect_turn_order and color != self.player:
            return []

        moves = []

        # four cardinal directions
        directions = [(-1,0),(1,0),(0,1),(0,-1)]
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            while 0 <= rr < board_dimensions and 0 <= cc < board_dimensions:
                target = self.board[rr][cc]
                if target == "--":
                    moves.append(self._mk_move((r,c),(rr,cc)))
                else:
                    if target[0] != color:
                        moves.append(self._mk_move((r,c),(rr,cc)))
                    break
                rr += dr
                cc += dc
        return moves
       
    def king_moves(self: "GameState", start: Sq, board_dimensions: int = 8, respect_turn_order: bool = True, **kwargs) -> list[Move]:
        """
        Generate pseudo-legal king moves from a given square.

        Parameters
        ----------
        start : (int, int)
            King starting square.
        board_dimensions : int, optional (default=8)
            Size of the board (assumed square).
        respect_turn_order : bool, optional (default=True)
            If True, return no moves unless the king color matches `self.player`.
            If False, generate moves regardless of whose turn it is.

        Returns
        -------
        list[Move]
            Pseudo-legal king moves to adjacent squares, including captures.
            Moves are not filtered for king safety.

        Notes
        -----
        - Castling is not generated here (to be added later).
        - Any extra keyword arguments are ignored (present for a uniform dispatcher
        interface).
        """
        r, c = start              # unpack start square
        piece = self.board[r][c]  # get piece at start square
        # throw error if piece is not king
        if piece[1] != "K" and piece != "--":
            raise ValueError("king_moves called on non-king piece")
        # if empty square, return no moves
        if piece == "--":
            return []
        color = piece[0]  # get color of king
        # if turn order is respected and king color is not current player, return no moves
        if respect_turn_order and color != self.player:
            return []

        moves = []

        is_attack = bool(kwargs.get("is_attack", False))

        # all adjacent squares
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                # skip the square the king is on
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < board_dimensions and 0 <= cc < board_dimensions:
                    target = self.board[rr][cc]
                    if target == "--" or target[0] != color:
                        moves.append(self._mk_move((r,c),(rr,cc)))

        # castling (only when generating moves, not attacks)
        if not is_attack and hasattr(self, "castling_rights"):
            enemy = "b" if color == "w" else "w"

            # only from the home square
            if (color == "w" and (r, c) == (7, 4)) or (color == "b" and (r, c) == (0, 4)):

                # king-side: e1->g1 / e8->g8, rook h-file -> f-file
                key_k = color + "K"
                if self.castling_rights.get(key_k, False):
                    row = 7 if color == "w" else 0
                    if self.board[row][5] == "--" and self.board[row][6] == "--" and self.board[row][7] == color + "R":
                        if (
                            not self.is_checked((row, 4), enemy)
                            and not self.is_checked((row, 5), enemy)
                            and not self.is_checked((row, 6), enemy)
                        ):
                            moves.append(
                                Move(
                                    start=(row, 4),
                                    end=(row, 6),
                                    piece_moved=color + "K",
                                    piece_captured="--",
                                    is_castle=True,
                                    rook_start=(row, 7),
                                    rook_end=(row, 5),
                                )
                            )

                # queen-side: e1->c1 / e8->c8, rook a-file -> d-file
                key_q = color + "Q"
                if self.castling_rights.get(key_q, False):
                    row = 7 if color == "w" else 0
                    if (
                        self.board[row][1] == "--"
                        and self.board[row][2] == "--"
                        and self.board[row][3] == "--"
                        and self.board[row][0] == color + "R"
                    ):
                        if (
                            not self.is_checked((row, 4), enemy)
                            and not self.is_checked((row, 3), enemy)
                            and not self.is_checked((row, 2), enemy)
                        ):
                            moves.append(
                                Move(
                                    start=(row, 4),
                                    end=(row, 2),
                                    piece_moved=color + "K",
                                    piece_captured="--",
                                    is_castle=True,
                                    rook_start=(row, 0),
                                    rook_end=(row, 3),
                                )
                            )
        return moves

    def bishop_moves(self: "GameState", start: Sq, board_dimensions: int = 8, respect_turn_order: bool = True, **kwargs) -> list[Move]:
        """
        Generate pseudo-legal bishop moves from a given square.

        Parameters
        ----------
        start : (int, int)
            Bishop starting square.
        board_dimensions : int, optional (default=8)
            Size of the board (assumed square).
        respect_turn_order : bool, optional (default=True)
            If True, return no moves unless the bishop color matches `self.player`.
            If False, generate moves regardless of whose turn it is.

        Returns
        -------
        list[Move]
            Pseudo-legal bishop moves along diagonals (captures included).
            Moves are not filtered for king safety.

        Notes
        -----
        Any extra keyword arguments are ignored (present for a uniform dispatcher
        interface).
        """
        r, c = start              # unpack start square
        piece = self.board[r][c]  # get piece at start square
        # throw error if piece is not bishop
        if piece[1] != "B" and piece != "--":
            raise ValueError("bishop_moves called on non-bishop piece")
        # if empty square, return no moves
        if piece == "--":
            return []
        color = piece[0]    # get color of bishop
        # if turn order is respected and bishop color is not current player, return no moves
        if respect_turn_order and color != self.player:
            return []

        moves = []

        # four diagonal directions
        directions = [(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            while 0 <= rr < board_dimensions and 0 <= cc < board_dimensions:
                target = self.board[rr][cc]
                if target == "--":
                    moves.append(self._mk_move((r,c),(rr,cc)))
                else:
                    if target[0] != color:
                        moves.append(self._mk_move((r,c),(rr,cc)))
                    break
                rr += dr
                cc += dc
        return moves

    def queen_moves(self: "GameState", start: Sq, board_dimensions: int = 8, respect_turn_order: bool = True, **kwargs) -> list[Move]:
        """
        Generate pseudo-legal queen moves from a given square.

        Parameters
        ----------
        start : (int, int)
            Queen starting square.
        board_dimensions : int, optional (default=8)
            Size of the board (assumed square).
        respect_turn_order : bool, optional (default=True)
            If True, return no moves unless the queen color matches `self.player`.
            If False, generate moves regardless of whose turn it is.

        Returns
        -------
        list[Move]
            Pseudo-legal queen moves (rook-like + bishop-like), including captures.
            Moves are not filtered for king safety.

        Notes
        -----
        Any extra keyword arguments are ignored (present for a uniform dispatcher
        interface).
        """
        r, c = start                # unpack start square
        piece = self.board[r][c]    # get piece at start square

        # if empty square, return no moves
        if piece == "--":
            return []
        
        # throw error if piece is not queen
        if piece[1] != "Q":
            raise ValueError("queen_moves called on non-queen piece")

        color = piece[0]   # get color of queen
        # if turn order is respected and queen color is not current player, return no moves
        if respect_turn_order and color != self.player:
            return []

        moves = []

        # Rook-like directions
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in rook_dirs:
            rr, cc = r + dr, c + dc
            while 0 <= rr < board_dimensions and 0 <= cc < board_dimensions:
                target = self.board[rr][cc]
                if target == "--":
                    moves.append(self._mk_move((r, c), (rr, cc)))
                else:
                    if target[0] != color:
                        moves.append(self._mk_move((r, c), (rr, cc)))
                    break
                rr += dr
                cc += dc

        # Bishop-like directions
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in bishop_dirs:
            rr, cc = r + dr, c + dc
            while 0 <= rr < board_dimensions and 0 <= cc < board_dimensions:
                target = self.board[rr][cc]
                if target == "--":
                    moves.append(self._mk_move((r, c), (rr, cc)))
                else:
                    if target[0] != color:
                        moves.append(self._mk_move((r, c), (rr, cc)))
                    break
                rr += dr
                cc += dc

        return moves

    def knight_moves(self: "GameState", start: Sq, board_dimensions: int = 8, respect_turn_order: bool = True, **kwargs) -> list[Move]:
        """
        Generate pseudo-legal knight moves from a given square.

        Parameters
        ----------
        start : (int, int)
            Knight starting square.
        board_dimensions : int, optional (default=8)
            Size of the board (assumed square).
        respect_turn_order : bool, optional (default=True)
            If True, return no moves unless the knight color matches `self.player`.
            If False, generate moves regardless of whose turn it is.

        Returns
        -------
        list[Move]
            Pseudo-legal knight moves (L-jumps), including captures.
            Moves are not filtered for king safety.

        Notes
        -----
        Any extra keyword arguments are ignored (present for a uniform dispatcher
        interface).
        """
        r, c = start                # unpack start square
        piece = self.board[r][c]    # get piece at start square
        # throw error if piece is not knight
        if piece[1] != "N" and piece != "--":
            raise ValueError("knight_moves called on non-knight piece")
        # if empty square, return no moves
        if piece == "--":
            return []
        color = piece[0]    # get color of knight
        # if turn order is respected and knight color is not current player, return no moves
        if respect_turn_order and color != self.player:
            return []

        moves = []

        # all possible knight jumps
        jumps = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
        for dr, dc in jumps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < board_dimensions and 0 <= cc < board_dimensions:
                target = self.board[rr][cc]
                if target == "--" or target[0] != color:
                    moves.append(self._mk_move((r,c),(rr,cc)))
        return moves

    def get_available_moves(self: "GameState", coord: Sq, as_moves: bool = True, **kwargs) -> list[Move] | list[UISq]:
        """
        Return pseudo-legal moves for the piece located at `coord`.

        Parameters
        ----------
        as_moves : bool
            If True, return Move objects.
            If False, return destination squares only.
        """
        MOVE_FUNCS = {
            "P": "pawn_moves",
            "B": "bishop_moves",
            "Q": "queen_moves",
            "R": "rook_moves",
            "K": "king_moves",
            "N": "knight_moves",
        }

        row, col = coord
        piece = self.board[row][col]
        if piece == "--":
            return []

        func = getattr(GameState, MOVE_FUNCS[piece[1]])
        moves = func(self, coord, **kwargs)

        if as_moves:
            return moves
        return [list(mv.end) for mv in moves]
    
    def iter_piece_squares(self: "GameState", color: str) -> Iterable[Sq]:
        """
        Yield coordinates of all pieces of a given color on the board.
        """
        for r in range(self.board_dimension):
            for c in range(self.board_dimension):
                if self.board[r][c] != "--" and self.board[r][c][0] == color:
                    yield (r, c)

    def iter_available_moves(self: "GameState", color: str, *, is_attack: bool) -> Iterable[Move]:
        """
        Yield pseudo-legal moves for all pieces of a given color.

        Parameters
        ----------
        is_attack : bool
            If True, generate attack maps instead of movement options.
        """
        for sq in self.iter_piece_squares(color):
            # respect_turn_order=False because we are explicitly choosing the color
            for mv in self.get_available_moves(sq,
                                            respect_turn_order=False,
                                            is_attack=is_attack,
                                            as_moves=True):
                yield mv

    def get_all_available(self: "GameState", color: str, is_attack: bool, as_moves: bool = True) -> list[Move] | list[UISq]:
        """
        Return all squares or moves available to a given color.

        Parameters
        ----------
        color : str
            Either "w" or "b", the color whose moves or attacks are requested.

        is_attack : bool
            If True, return the attack map for this color (used for check detection).
            If False, return pseudo-legal move destinations.

        as_moves : bool, optional (default=True)
            If True, return a list of Move objects.
            If False, return a list of destination squares as [row, col].

        Returns
        -------
        list
            Either:
            - List[Move] if as_moves=True
            - List[[row, col]] if as_moves=False
        """

        # Generate all pseudo-legal Move objects for this color
        # respect_turn_order=False because the color is explicitly specified
        moves = list(
            self.iter_available_moves(color, is_attack=is_attack)
        )

        # If Move objects are requested, return them directly
        if as_moves:
            return moves

        # Otherwise, project down to destination squares for UI / legacy code
        return [list(mv.end) for mv in moves]

    def is_checked(self: "GameState", king_coord: Sq, attacker_color: str) -> bool:
        """
        Determine whether a king is in check.

        Returns True if the square `king_coord` is attacked by `attacker_color`.
        """
        king_sq = tuple(king_coord)
        for mv in self.iter_available_moves(attacker_color, is_attack=True):
            if mv.end == king_sq:
                return True
        return False

    def get_king_coord(self: "GameState", color: str) -> Optional[Sq]:
        """
        Return the coordinates of the king of the specified color.

        Returns None if the king is not found.
        """
        king_name = f"{color}K"
        for r in range(self.board_dimension):
            for c in range(self.board_dimension):
                if self.board[r][c] == king_name:
                    return (r, c)
        return None  # (optional) explicit failure case

    def get_legal_moves(self: "GameState", coord: Sq, as_moves: bool = True, respect_turn_order: bool = True) -> list[Move] | list[UISq]:
        """
        Return fully legal moves for the piece at `coord`.

        This method starts from pseudo-legal moves (piece motion + occupancy) and
        filters them by simulating each move and rejecting any move that leaves
        the moving side's king in check.

        Parameters
        ----------
        coord : (int, int)
            Coordinates of the piece to move.
        as_moves : bool, optional (default=True)
            If True, return a list of `Move` objects.
            If False, return a list of destination squares as [row, col] (UI-friendly).
        respect_turn_order : bool, optional (default=True)
            If True, only generate moves if the piece color matches `self.player`.
            If False, generate legal moves for the piece regardless of whose turn it is.

        Returns
        -------
        list
            Either:
            - list[Move] if as_moves=True
            - list[[row, col]] if as_moves=False
        """
        mover_color = self.player
        available = self.get_available_moves(coord,
                                            respect_turn_order=respect_turn_order,
                                            is_attack=False,
                                            as_moves=True)

        legal = []
        for mv in available:
            self.make_move(mv)
            attacker = self.player  # after make_move, self.player switched
            our_king = self.get_king_coord(mover_color)
            if not self.is_checked(our_king, attacker):
                legal.append(mv)
            self.undo_last_move()
        if as_moves:
            return legal
        else: 
            return [list(mv.end) for mv in legal]

    def is_legal_move(self: "GameState", move_pair: Tuple[UISq, UISq]) -> bool:
        """
        Determine whether a given move is legal for the current player.

        Parameters
        ----------
        move_pair : Tuple[UISq, UISq]
            A tuple containing the start and end coordinates of the move.

        Returns
        -------
        bool
            True if the move is legal, False otherwise.
        """
        start_ui, end_ui = move_pair
        start: Sq = tuple(start_ui)
        end: Sq = tuple(end_ui)

        for mv in self.get_legal_moves(start):
            if mv.end == end:
                return True
        return False

    def get_all_legal_moves(self: "GameState", color: str, as_moves: bool = True) -> list[Move] | list[UISq]:
        """
        Return all legal moves for `color`.

        Parameters
        ----------
        color : str
            "w" or "b"
        as_moves : bool
            If True, return List[Move].
            If False, return List[[row, col]] of destination squares (UI-friendly).

        Notes
        -----
        This function temporarily sets self.player = color so that the existing
        per-piece legality routines (which depend on self.player) work correctly.
        The original player is restored before returning.
        """
        saved_player = self.player
        self.player = color
        try:
            all_moves = []

            # Iterate over all pieces of this color and collect their legal moves
            for sq in self.iter_piece_squares(color):
                # IMPORTANT: use the Move-native function here
                all_moves.extend(self.get_legal_moves(sq, respect_turn_order=False))

            if as_moves:
                return all_moves

            # UI / legacy: return just destination squares
            return [list(mv.end) for mv in all_moves]

        finally:
            self.player = saved_player

    def is_checkmate(self: "GameState") -> bool:
        """True iff current player has no legal moves and is in check."""
        total_legal = len(self.get_all_legal_moves(self.player))
        king_coord = self.get_king_coord(self.player)
        in_check = self.is_checked(king_coord, self.opponent_color())
        return total_legal == 0 and in_check

    def is_stalemate(self: "GameState") -> bool:
        """True iff current player has no legal moves and is not in check."""
        total_legal = len(self.get_all_legal_moves(self.player))
        king_coord = self.get_king_coord(self.player)
        in_check = self.is_checked(king_coord, self.opponent_color())
        return total_legal == 0 and not in_check

