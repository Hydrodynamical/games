from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
from copy import deepcopy
import math
import numpy as np

from tqdm import trange

try:
    # When imported as a package module (e.g. `from scr.mcts import ...`)
    from .rl_interface import policy_distribution, critic_value
except ImportError:  # pragma: no cover
    # When executed with `chess/scr` on sys.path (e.g. `python main.py` from chess/scr)
    from rl_interface import policy_distribution, critic_value

# from chess_engine import GameState, Move
# from rl_interface import PolicyValueNet, policy_distribution, critic_value

@dataclass
class MCTSDiagnostics:
    sims: int = 0
    terminal_checkmate: int = 0
    terminal_stalemate: int = 0
    terminal_other: int = 0
    expanded_nonterminal: int = 0

    def add(self, other: "MCTSDiagnostics") -> None:
        self.sims += other.sims
        self.terminal_checkmate += other.terminal_checkmate
        self.terminal_stalemate += other.terminal_stalemate
        self.terminal_other += other.terminal_other
        self.expanded_nonterminal += other.expanded_nonterminal

    def terminal_total(self) -> int:
        return self.terminal_checkmate + self.terminal_stalemate + self.terminal_other

@dataclass
class EdgeStats:
    prior: float
    n: int = 0
    w: float = 0.0  # total value from current player's perspective

    @property
    def q(self) -> float:
        return 0.0 if self.n == 0 else self.w / self.n


class Node:
    def __init__(self, gs):
        self.gs = gs
        self.is_expanded = False
        self.children: Dict[object, "Node"] = {}
        self.edges: Dict[object, EdgeStats] = {}  # keyed by Move

    def expand(
        self,
        model,
        add_dirichlet_noise: bool = False,
        dir_alpha: float = 0.3,
        dir_epsilon: float = 0.25,
    ):
        # priors over legal moves
        priors = policy_distribution(model, self.gs)

        if add_dirichlet_noise and len(priors) > 0:
            moves = list(priors.keys())
            p = np.array([max(0.0, float(priors[m])) for m in moves], dtype=np.float64)

            # normalize (avoid weirdness if model returns non-normalized probs)
            s = p.sum()
            if s <= 0:
                p = np.ones_like(p) / len(p)
            else:
                p = p / s

            # Dirichlet noise
            noise = np.random.dirichlet([dir_alpha] * len(moves))

            p = (1.0 - dir_epsilon) * p + dir_epsilon * noise

            priors = {m: float(p[i]) for i, m in enumerate(moves)}

        for mv, prob in priors.items():
            self.edges[mv] = EdgeStats(prior=float(prob))

        self.is_expanded = True

    def is_terminal(self) -> bool:
        # draw by repetition / 50-move
        if hasattr(self.gs, "is_draw") and self.gs.is_draw():
            return True
        # checkmate/stalemate fall out of "no legal moves"
        moves = self.gs.get_all_legal_moves(self.gs.player, as_moves=True)
        return len(moves) == 0



def ucb_score(parent_visits: int, edge: EdgeStats, c_puct: float) -> float:
    # AlphaZero-style
    return edge.q + c_puct * edge.prior * math.sqrt(parent_visits + 1e-8) / (1 + edge.n)


def select_move(node: Node, c_puct: float) -> object:
    parent_n = sum(e.n for e in node.edges.values())
    best_mv, best_sc = None, -1e30
    for mv, e in node.edges.items():
        sc = ucb_score(parent_n, e, c_puct)
        if sc > best_sc:
            best_mv, best_sc = mv, sc
    return best_mv


def apply_move_copy(gs, mv):
    gs2 = deepcopy(gs)
    gs2.make_move(mv)
    return gs2


def mcts_search(
    root_gs,
    model,
    num_sims: int = 200,
    c_puct: float = 1.5,
    show_progress: bool = False,
    diagnostics: MCTSDiagnostics | None = None,   # NEW optional output
    add_root_dirichlet: bool = False,   # NEW
    dir_alpha: float = 0.3,             # NEW
    dir_epsilon: float = 0.25,          # NEW
):
    root = Node(deepcopy(root_gs))
    if not root.is_expanded:
        root.expand(
            model,
            add_dirichlet_noise=add_root_dirichlet,
            dir_alpha=dir_alpha,
            dir_epsilon=dir_epsilon,
        )

    sim_iter = trange(num_sims, desc="MCTS", leave=False) if show_progress else range(num_sims)
    
    local = MCTSDiagnostics() # NEW: local counters for this call

    for _ in sim_iter:
        local.sims += 1

        node = root
        path: List[tuple[Node, object]] = []

        # Selection
        while node.is_expanded and not node.is_terminal():
            mv = select_move(node, c_puct)
            path.append((node, mv))
            if mv not in node.children:
                node.children[mv] = Node(apply_move_copy(node.gs, mv))
            node = node.children[mv]

        # Evaluation / Expansion
        if node.is_terminal():
            # draw (repetition / 50-move)
            if hasattr(node.gs, "is_draw") and node.gs.is_draw():
                local.terminal_other += 1
                v = 0.0
            # checkmate
            if hasattr(node.gs, "is_checkmate") and node.gs.is_checkmate():
                local.terminal_checkmate += 1
                v = -1.0
            # stalemate
            elif hasattr(node.gs, "is_stalemate") and node.gs.is_stalemate():
                local.terminal_stalemate += 1
                v = 0.0
            # other (fallback)
            else:
                local.terminal_other += 1
                v = 0.0

        else:
            local.expanded_nonterminal += 1
            node.expand(model)
            v = critic_value(model, node.gs)

        # "Frozen" terminal updates via tqdm (no scrolling)
        if show_progress:
            try:
                if local.sims == 1 or local.sims % 10 == 0 or local.sims == num_sims:
                    sim_iter.set_postfix(
                        cm=local.terminal_checkmate,
                        sm=local.terminal_stalemate,
                        other=local.terminal_other,
                        exp=local.expanded_nonterminal,
                    )
            except Exception:
                pass

        # Backprop
        # v is from the perspective of the player to move at "node.gs".
        # As we go back up a ply, the player alternates, so flip sign each step.
        for parent, mv in reversed(path):
            v = -v # Fixed: flip sign at each step up
            e = parent.edges[mv]
            e.n += 1
            e.w += v
            # v = -v # Alternate perspective? Might be wrong...

    # NEW: accumulate into the passed-in object
    if diagnostics is not None:
        diagnostics.add(local)
    # NOTE: For in-place progress output, call with show_progress=True.

    # Return improved move distribution proportional to visit counts
    visits = {mv: e.n for mv, e in root.edges.items()}
    total = sum(visits.values())
    pi = {mv: (n / total if total > 0 else 0.0) for mv, n in visits.items()}
    return pi

# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500