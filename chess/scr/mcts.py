"""PUCT Monte Carlo tree search with optional neural policy/value evaluation.

The default evaluator uses material and simple piece placement, so playing chess
needs only Python's standard library. Supplying a PolicyValueNet enables batched
neural leaf evaluation without changing the search interface.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
from typing import Dict, List, Optional, Tuple


@dataclass
class MCTSDiagnostics:
    sims: int = 0
    terminal_checkmate: int = 0
    terminal_stalemate: int = 0
    terminal_other: int = 0
    expanded_nonterminal: int = 0

    def add(self, other: "MCTSDiagnostics") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def terminal_total(self) -> int:
        return self.terminal_checkmate + self.terminal_stalemate + self.terminal_other


@dataclass
class EdgeStats:
    prior: float
    n: int = 0
    w: float = 0.0  # total value from the parent player's perspective

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}


def evaluate_position(gs) -> float:
    """Bounded static value from the side-to-move's perspective.

    This is a useful starting evaluator, not a trained model or a tactical
    oracle. Search supplies the look-ahead and exact terminal outcomes.
    """
    score = 0.0
    non_pawn_material = sum(
        PIECE_VALUES[p[1]] for row in gs.board for p in row
        if p != "--" and p[1] not in ("P", "K")
    )
    for r, row in enumerate(gs.board):
        for c, piece in enumerate(row):
            if piece == "--":
                continue
            color, kind = piece
            advancement = (6 - r) if color == "w" else (r - 1)
            center = 7 - abs(3.5 - r) - abs(3.5 - c)
            placement = 0.0
            if kind == "P":
                placement = 8 * advancement + 2 * center
            elif kind == "N":
                placement = 8 * center
            elif kind == "B":
                placement = 4 * center
            elif kind == "R":
                placement = 2 * advancement
            elif kind == "K":
                if non_pawn_material <= 2000:
                    placement = 5 * center
                elif (r == (7 if color == "w" else 0)) and c in (2, 6):
                    placement = 25
            score += (1 if color == "w" else -1) * (PIECE_VALUES[kind] + placement)
    if gs.player == "b":
        score = -score
    # A static evaluation never equals a proven win/loss.
    return max(-0.99, min(0.99, math.tanh(score / 600.0)))


def _normalise(weights: List[float]) -> List[float]:
    weights = [max(0.0, float(p)) if math.isfinite(float(p)) else 0.0 for p in weights]
    total = sum(weights)
    if not weights:
        return []
    return [p / total for p in weights] if total else [1.0 / len(weights)] * len(weights)


def _heuristic_priors(moves) -> Dict[object, float]:
    weights = []
    for mv in moves:
        captured = getattr(mv, "piece_captured", "--")
        promotion = getattr(mv, "promotion", "")
        weight = 1.0
        if captured != "--":
            weight += PIECE_VALUES.get(captured[-1], 0) / 400.0
        if promotion:
            weight += PIECE_VALUES.get(promotion[-1].upper(), 0) / 400.0
        if getattr(mv, "is_castle", False):
            weight += 0.5
        weights.append(weight)
    return dict(zip(moves, _normalise(weights)))


def _neural_helpers():
    # These optional dependencies are deliberately absent from the default path.
    try:
        from .rl_interface import encode_gamestate, move_to_action, policy_distribution
    except ImportError:
        from rl_interface import encode_gamestate, move_to_action, policy_distribution
    return encode_gamestate, move_to_action, policy_distribution


def _state_signature(gs):
    """Board identity alone is insufficient for repetition/50-move draws."""
    return (
        gs.position_key(),
        getattr(gs, "halfmove_clock", 0),
        frozenset(getattr(gs, "position_counts", {}).items()),
    )


class Node:
    def __init__(self, gs):
        self.gs = gs
        self.is_expanded = False
        self.children: Dict[object, Node] = {}
        self.edges: Dict[object, EdgeStats] = {}
        self._base_priors: Dict[object, float] = {}
        self._legal_moves_cache: Optional[List[object]] = None
        self._is_terminal_cache: Optional[bool] = None
        self._terminal_value_cache: Optional[float] = None
        self.terminal_reason: Optional[str] = None
        self._model_id: Optional[int] = None

    def legal_moves(self) -> List[object]:
        if self._legal_moves_cache is None:
            self._legal_moves_cache = self.gs.get_all_legal_moves(self.gs.player, as_moves=True)
        return self._legal_moves_cache

    def _set_priors(self, priors) -> None:
        moves = self.legal_moves()
        probabilities = _normalise([priors.get(mv, 0.0) for mv in moves])
        self._base_priors = dict(zip(moves, probabilities))
        self.edges = {mv: EdgeStats(p) for mv, p in self._base_priors.items()}
        self.is_expanded = True

    def root_noise(self, enabled: bool, alpha: float, epsilon: float) -> None:
        """Fresh root noise each move, without repeatedly mixing old noise."""
        moves = list(self.edges)
        if not moves:
            return
        noise = (_normalise([random.gammavariate(alpha, 1.0) for _ in moves])
                 if enabled else [0.0] * len(moves))
        for mv, sample in zip(moves, noise):
            base = self._base_priors[mv]
            self.edges[mv].prior = ((1 - epsilon) * base + epsilon * sample) if enabled else base

    def expand(self, model=None, add_dirichlet_noise=False, dir_alpha=0.3, dir_epsilon=0.25):
        if self.is_terminal():
            self.is_expanded = True
            return
        moves = self.legal_moves()
        if model is None:
            priors = _heuristic_priors(moves)
        else:
            _, _, policy_distribution = _neural_helpers()
            priors = policy_distribution(model, self.gs, legal_moves=moves)
        self._set_priors(priors)
        self._model_id = id(model)
        self.root_noise(add_dirichlet_noise, dir_alpha, dir_epsilon)

    def is_terminal(self) -> bool:
        if self._is_terminal_cache is None:
            # Checkmate takes precedence over a draw counter reached on that move.
            if not self.legal_moves():
                mover = self.gs.player
                king = self.gs.get_king_coord(mover)
                opp = "b" if mover == "w" else "w"
                in_check = king is not None and self.gs.is_checked(king, opp)
                self._terminal_value_cache = -1.0 if in_check else 0.0
                self.terminal_reason = "checkmate" if in_check else "stalemate"
                self._is_terminal_cache = True
            elif hasattr(self.gs, "is_draw") and self.gs.is_draw():
                self._terminal_value_cache = 0.0
                self.terminal_reason = "draw"
                self._is_terminal_cache = True
            else:
                self._is_terminal_cache = False
        return self._is_terminal_cache

    def terminal_value(self) -> float:
        """Exact result from the player-to-move's perspective."""
        if not self.is_terminal():
            raise ValueError("terminal_value requires a terminal position")
        return self._terminal_value_cache


def ucb_score(parent_visits: int, edge: EdgeStats, c_puct: float) -> float:
    return edge.q + c_puct * edge.prior * math.sqrt(parent_visits + 1) / (1 + edge.n)


def select_move(node: Node, c_puct: float) -> object:
    # Try every legal action once, in prior order, before exploiting its value.
    # In particular an immediate mate cannot be starved by a misleading prior.
    unvisited = [mv for mv, edge in node.edges.items() if edge.n == 0]
    if unvisited:
        return max(unvisited, key=lambda mv: node.edges[mv].prior)
    visits = sum(edge.n for edge in node.edges.values())
    return max(node.edges, key=lambda mv: ucb_score(visits, node.edges[mv], c_puct))


def apply_move_copy(gs, mv):
    result = gs.fast_copy_for_search() if hasattr(gs, "fast_copy_for_search") else deepcopy(gs)
    result.make_move(mv)
    return result


def best_move(distribution, root: Node | None = None):
    """Choose by visits/probability, then value and prior; return None if terminal."""
    if not distribution:
        return None

    def score(mv):
        edge = root.edges.get(mv) if root is not None else None
        return (distribution[mv], edge.q if edge else 0.0, edge.prior if edge else 0.0)

    return max(distribution, key=score)


def _expand_from_logits(node, logits_row, *, add_dirichlet_noise, dir_alpha, dir_epsilon):
    import torch
    _, move_to_action, _ = _neural_helpers()
    moves = node.legal_moves()
    actions = torch.tensor([move_to_action(mv) for mv in moves],
                           device=logits_row.device, dtype=torch.long)
    probabilities = torch.softmax(logits_row[actions], dim=0).detach().cpu().tolist()
    node._set_priors(dict(zip(moves, probabilities)))
    node.root_noise(add_dirichlet_noise, dir_alpha, dir_epsilon)


def mcts_search(
    root_gs,
    model=None,
    num_sims: int = 200,
    c_puct: float = 1.5,
    show_progress: bool = False,
    diagnostics: MCTSDiagnostics | None = None,
    add_root_dirichlet: bool = False,
    dir_alpha: float = 0.3,
    dir_epsilon: float = 0.25,
    root: Node | None = None,
    return_root: bool = False,
    batch_size: int = 16,
    time_limit: float | None = None,
    stop_event=None,
) -> Dict[object, float] | Tuple[Dict[object, float], Node]:
    """Return legal move probabilities from completed MCTS visit counts.

    ``num_sims`` bounds iterations; ``time_limit`` is an optional seconds budget.
    ``stop_event`` may be a threading.Event. Cancellation is checked between
    simulations; move generation and an in-flight neural batch finish first.
    A zero budget returns normalized priors, and a terminal root returns {}.
    Input positions are never changed. Reused roots must have identical board,
    side, special rights, halfmove clock, repetition history, and evaluator.
    """
    if not isinstance(num_sims, int) or num_sims < 0:
        raise ValueError("num_sims must be a nonnegative integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(c_puct) or c_puct < 0:
        raise ValueError("c_puct must be finite and nonnegative")
    if not math.isfinite(dir_alpha) or dir_alpha <= 0:
        raise ValueError("dir_alpha must be finite and positive")
    if not math.isfinite(dir_epsilon) or not 0 <= dir_epsilon <= 1:
        raise ValueError("dir_epsilon must lie in [0, 1]")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative")

    deadline = None if time_limit is None else time.monotonic() + time_limit

    def stopped():
        return ((stop_event is not None and stop_event.is_set())
                or (deadline is not None and time.monotonic() >= deadline))

    if root is None or _state_signature(root.gs) != _state_signature(root_gs) or (
        root.is_expanded and root._model_id != id(model)
    ):
        copied = root_gs.fast_copy_for_search() if hasattr(root_gs, "fast_copy_for_search") else deepcopy(root_gs)
        root = Node(copied)
    local = MCTSDiagnostics()
    if root.is_terminal():
        return ({}, root) if return_root else {}
    if not root.is_expanded:
        # Even an already-expired budget needs a legal fallback; avoid neural
        # inference in that case, and do not mark those priors as model-derived.
        root.expand(None if stopped() else model)
    root.root_noise(add_root_dirichlet, dir_alpha, dir_epsilon)

    pending = []
    pending_ids = set()

    def backprop(path, value, reserved=False):
        for parent, mv in reversed(path):
            value = -value
            edge = parent.edges[mv]
            if reserved:
                edge.w += 1.0 + value  # remove the virtual loss
            else:
                edge.n += 1
                edge.w += value

    def flush_batch():
        if not pending:
            return
        completed = 0
        try:
            import torch
            encode_gamestate, _, _ = _neural_helpers()
            model.eval()
            device = next(model.parameters()).device
            states = torch.stack([encode_gamestate(node.gs) for node, _ in pending]).to(device)
            with torch.inference_mode():
                logits, values = model(states)
            if not torch.is_tensor(logits) or logits.ndim != 2 or logits.shape[0] != len(pending):
                raise ValueError("neural policy must have shape [batch, actions]")
            if not torch.is_tensor(values) or values.numel() != len(pending):
                raise ValueError("neural value must contain one value per position")
            values = values.reshape(-1).detach().cpu().tolist()
            for index, (node, path) in enumerate(pending):
                _expand_from_logits(node, logits[index], add_dirichlet_noise=False,
                                    dir_alpha=dir_alpha, dir_epsilon=dir_epsilon)
                node._model_id = id(model)
                value = float(values[index])
                if not math.isfinite(value):
                    value = 0.0
                backprop(path, max(-1.0, min(1.0, value)), reserved=True)
                local.expanded_nonterminal += 1
                completed += 1
        except BaseException:
            # Do not leave virtual visits/losses in a reusable tree on failure.
            # Already completed paths must not be rolled back.
            for node, path in pending[completed:]:
                node.edges.clear()
                node._base_priors.clear()
                node.is_expanded = False
                node._model_id = None
                for parent, mv in path:
                    parent.edges[mv].n -= 1
                    parent.edges[mv].w += 1.0
            raise
        finally:
            pending.clear()
            pending_ids.clear()

    iterations = range(num_sims)
    progress = None
    if show_progress:
        try:
            from tqdm import tqdm
            progress = tqdm(iterations, desc="MCTS", leave=False)
            iterations = progress
        except ImportError:
            pass
    try:
        for _ in iterations:
            if stopped():
                break
            node = root
            path = []
            while node.is_expanded and not node.is_terminal():
                mv = select_move(node, c_puct)
                path.append((node, mv))
                if mv not in node.children:
                    node.children[mv] = Node(apply_move_copy(node.gs, mv))
                node = node.children[mv]
                # A pending leaf must be evaluated before it can be selected
                # again; this avoids counting duplicate NN evaluations as work.
                if id(node) in pending_ids:
                    flush_batch()
            local.sims += 1
            if node.is_terminal():
                value = node.terminal_value()
                if node.terminal_reason == "checkmate":
                    local.terminal_checkmate += 1
                elif node.terminal_reason == "stalemate":
                    local.terminal_stalemate += 1
                else:
                    local.terminal_other += 1
                backprop(path, value)
            elif model is None:
                node.expand()
                local.expanded_nonterminal += 1
                backprop(path, evaluate_position(node.gs))
            else:
                pending.append((node, path))
                pending_ids.add(id(node))
                for parent, mv in path:
                    parent.edges[mv].n += 1
                    parent.edges[mv].w -= 1.0
                if len(pending) >= batch_size:
                    flush_batch()
        flush_batch()
    finally:
        if progress is not None:
            progress.close()
        if diagnostics is not None:
            diagnostics.add(local)

    total = sum(edge.n for edge in root.edges.values())
    if total:
        probabilities = {mv: edge.n / total for mv, edge in root.edges.items()}
    else:
        probabilities = {mv: edge.prior for mv, edge in root.edges.items()}
    return (probabilities, root) if return_root else probabilities
