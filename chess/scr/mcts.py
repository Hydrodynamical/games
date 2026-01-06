from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
from copy import deepcopy
import math

from tqdm import trange

from rl_interface import policy_distribution, critic_value

import random

import torch

# from chess_engine import GameState, Move
# from rl_interface import PolicyValueNet, policy_distribution, critic_value

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

    def expand(self, model):
        # priors over legal moves
        priors = policy_distribution(model, self.gs)
        for mv, p in priors.items():
            self.edges[mv] = EdgeStats(prior=p)
        self.is_expanded = True

    def is_terminal(self) -> bool:
        # simple terminal: no legal moves
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
):
    root = Node(deepcopy(root_gs))
    if not root.is_expanded:
        root.expand(model)

    sim_iter = trange(num_sims, desc="MCTS", leave=False) if show_progress else range(num_sims)

    for _ in sim_iter:
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
            # crude terminal value: current player has no legal moves => losing-ish
            v = -1.0
        else:
            node.expand(model)
            v = critic_value(model, node.gs)

        # Backprop
        # v is from the perspective of the player to move at "node.gs".
        # As we go back up a ply, the player alternates, so flip sign each step.
        for parent, mv in reversed(path):
            e = parent.edges[mv]
            e.n += 1
            e.w += v
            v = -v

    # Return improved move distribution proportional to visit counts
    visits = {mv: e.n for mv, e in root.edges.items()}
    total = sum(visits.values())
    pi = {mv: (n / total if total > 0 else 0.0) for mv, n in visits.items()}
    return pi

# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500