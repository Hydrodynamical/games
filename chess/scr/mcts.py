from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from typing import Dict, Optional, List
from copy import deepcopy
import math
import numpy as np

from tqdm import trange
import torch
import torch.nn.functional as F

try:
    # When imported as a package module (e.g. `from scr.mcts import ...`)
    from .rl_interface import policy_distribution, critic_value, encode_gamestate, move_to_action
except ImportError:  # pragma: no cover
    # When executed with `chess/scr` on sys.path (e.g. `python main.py` from chess/scr)
    from rl_interface import policy_distribution, critic_value, encode_gamestate, move_to_action

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

        # --- caches to avoid repeated expensive move generation ---
        self._legal_moves_cache: Optional[List[object]] = None
        self._is_terminal_cache: Optional[bool] = None
        self._terminal_value_cache: Optional[float] = None  # from player-to-move perspective

    def legal_moves(self) -> List[object]:
        """Cached list of legal moves for gs.player at this node."""
        if self._legal_moves_cache is None:
            self._legal_moves_cache = self.gs.get_all_legal_moves(self.gs.player, as_moves=True)
        return self._legal_moves_cache

    def expand(
        self,
        model,
        add_dirichlet_noise: bool = False,
        dir_alpha: float = 0.3,
        dir_epsilon: float = 0.25,
    ):
        # priors over legal moves (reuse cached legal moves to avoid recomputing)
        lm = self.legal_moves()
        # If terminal, there's nothing to expand
        if len(lm) == 0:
            self._is_terminal_cache = True
            self.is_expanded = True
            return

        priors = policy_distribution(model, self.gs, legal_moves=lm)

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
        if self._is_terminal_cache is not None:
            return self._is_terminal_cache

        # draw by repetition / 50-move
        if hasattr(self.gs, "is_draw") and self.gs.is_draw():
            self._is_terminal_cache = True
            return True

        # checkmate/stalemate fall out of "no legal moves"
        self._is_terminal_cache = (len(self.legal_moves()) == 0)
        return self._is_terminal_cache

    def terminal_value(self) -> float:
        """Terminal node value from perspective of player to move.

        - draw => 0
        - checkmate => -1
        - stalemate => 0
        """
        if self._terminal_value_cache is not None:
            return self._terminal_value_cache

        if hasattr(self.gs, "is_draw") and self.gs.is_draw():
            self._terminal_value_cache = 0.0
            return 0.0

        mover = self.gs.player
        opp = self.gs.opponent_color() if hasattr(self.gs, "opponent_color") else ("b" if mover == "w" else "w")

        king_sq = self.gs.get_king_coord(mover) if hasattr(self.gs, "get_king_coord") else None
        in_check = False
        if king_sq is not None and hasattr(self.gs, "is_checked"):
            in_check = self.gs.is_checked(king_sq, opp)

        self._terminal_value_cache = (-1.0 if in_check else 0.0)
        return self._terminal_value_cache



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
    if hasattr(gs, "fast_copy_for_search"):
        gs2 = gs.fast_copy_for_search()
    else:
        gs2 = deepcopy(gs)
    gs2.make_move(mv)
    return gs2


def _expand_from_logits(
    node: Node,
    logits_row: torch.Tensor,  # shape [4096] on device
    *,
    add_dirichlet_noise: bool,
    dir_alpha: float,
    dir_epsilon: float,
) -> None:
    """Expand a node using precomputed policy logits (batched NN inference)."""
    lm = node.legal_moves()
    if len(lm) == 0:
        node._is_terminal_cache = True
        node.is_expanded = True
        return

    legal_actions = [move_to_action(mv) for mv in lm]
    a_t = torch.tensor(legal_actions, device=logits_row.device, dtype=torch.long)
    legal_logits = logits_row[a_t]  # [num_legal]
    legal_probs = F.softmax(legal_logits, dim=0)

    # Move to CPU as a small vector (minimizes sync/copy)
    p = legal_probs.detach().cpu().numpy()

    if add_dirichlet_noise and len(lm) > 0:
        pn = np.array([max(0.0, float(x)) for x in p], dtype=np.float64)
        s = pn.sum()
        if s <= 0:
            pn = np.ones_like(pn) / len(pn)
        else:
            pn = pn / s
        noise = np.random.dirichlet([dir_alpha] * len(lm))
        pn = (1.0 - dir_epsilon) * pn + dir_epsilon * noise
        p = pn

    node.edges = {mv: EdgeStats(prior=float(p[i])) for i, mv in enumerate(lm)}
    node.is_expanded = True


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
    root: Node | None = None,           # NEW: reuse tree root across plies
    return_root: bool = False,          # NEW: optionally return root node
    batch_size: int = 16,               # NEW: batch leaf NN evals (esp. good on MPS)
) -> Dict[object, float] | Tuple[Dict[object, float], Node]:
    # Reuse provided root if available; otherwise start fresh from root_gs.
    if root is None:
        if hasattr(root_gs, "fast_copy_for_search"):
            root = Node(root_gs.fast_copy_for_search())
        else:
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

    # Accumulate nonterminal leaves for batched NN eval
    pending_nodes: List[Node] = []
    pending_paths: List[List[tuple[Node, object]]] = []

    def flush_batch() -> None:
        """Evaluate pending leaves in one forward pass, expand them, then backprop."""
        nonlocal pending_nodes, pending_paths
        if not pending_nodes:
            return

        model.eval()
        device = next(model.parameters()).device

        xs = torch.stack([encode_gamestate(n.gs) for n in pending_nodes], dim=0).to(device)
        with torch.inference_mode():
            logits, values = model(xs)  # logits [B,4096], values [B,1]

        values = values.squeeze(1)  # [B]

        for i, (node, path) in enumerate(zip(pending_nodes, pending_paths)):
            if not node.is_expanded:
                local.expanded_nonterminal += 1
                _expand_from_logits(
                    node,
                    logits[i],
                    add_dirichlet_noise=False,
                    dir_alpha=dir_alpha,
                    dir_epsilon=dir_epsilon,
                )

            v = float(values[i].detach().item())

            # Backprop values ONLY.
            # NOTE: visit counts were already incremented as "virtual visits" during selection,
            # so do NOT increment e.n here again.
            for parent, mv in reversed(path):
                v = -v
                e = parent.edges[mv]
                e.w += v

        pending_nodes = []
        pending_paths = []

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
            v = node.terminal_value()
            if v < 0:
                local.terminal_checkmate += 1
            elif v == 0.0:
                local.terminal_stalemate += 1
            else:
                local.terminal_other += 1

            # Backprop terminal immediately (no NN needed)
            for parent, mv in reversed(path):
                v = -v
                e = parent.edges[mv]
                e.n += 1
                e.w += v

        else:
            pending_nodes.append(node)
            pending_paths.append(path)

            # --- Virtual visits ---
            # Immediately increment visit counts along the selected path so UCB changes
            # for subsequent sims in the same batch. This prevents the search from collapsing
            # onto a single path when backprop is deferred.
            for parent, mv in path:
                parent.edges[mv].n += 1

            if len(pending_nodes) >= max(1, int(batch_size)):
                flush_batch()

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

        # NOTE: Backprop for nonterminal leaves is deferred to flush_batch().

    # Flush any leftover pending leaves
    flush_batch()

    # NEW: accumulate into the passed-in object
    if diagnostics is not None:
        diagnostics.add(local)
    # NOTE: For in-place progress output, call with show_progress=True.

    # Return improved move distribution proportional to visit counts
    visits = {mv: e.n for mv, e in root.edges.items()}
    total = sum(visits.values())
    pi = {mv: (n / total if total > 0 else 0.0) for mv, n in visits.items()}
    if return_root:
        return pi, root
    return pi

# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500