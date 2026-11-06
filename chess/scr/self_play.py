"""Generate policy targets by MCTS self-play and neutral draw value targets."""
from __future__ import annotations

from copy import deepcopy
import warnings

import torch

try:
    from .rl_interface import encode_gamestate, move_to_action, NUM_ACTIONS, pick_move_from_dist
    from .mcts import mcts_search, MCTSDiagnostics, best_move
except ImportError:
    from rl_interface import encode_gamestate, move_to_action, NUM_ACTIONS, pick_move_from_dist
    from mcts import mcts_search, MCTSDiagnostics, best_move


def terminal_result(gs):
    """White's result, or None while playing. Checkmate takes precedence."""
    if gs.is_checkmate():
        return -1.0 if gs.player == "w" else 1.0
    if gs.is_stalemate() or gs.is_draw():
        return 0.0
    return None


def play_self_game(
    gs, model, num_mcts_sims: int = 30, temperature: float = 1.0,
    max_moves: int = 200, show_progress: bool = False,
    trunc_draw_penalty: float = 0.0, draw_penalty: float = 0.0,
    diagnostics_per_ply: bool = False, add_root_dirichlet: bool = True,
    dir_alpha: float = 0.3, dir_epsilon: float = 0.25,
):
    """Return (state, visit-policy, mover) samples and White's result.

    Draws and games stopped at max_moves are scored zero. The legacy penalty
    arguments are accepted but ignored: last-mover penalties create incorrect
    zero-sum training labels. Temperature controls played moves; training policy
    targets remain the normalized MCTS visit counts.
    """
    if max_moves < 0 or num_mcts_sims < 0:
        raise ValueError("Move and simulation budgets must be nonnegative.")
    # Validate temperature even for an initially terminal game.
    from math import isfinite
    if not isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    if trunc_draw_penalty or draw_penalty:
        warnings.warn("Draw penalties are ignored; chess draws have value zero.", DeprecationWarning, stacklevel=2)
    gs = deepcopy(gs)
    data, root_node = [], None
    result = terminal_result(gs)
    if result is not None:
        return data, result
    plies = range(max_moves)
    if show_progress:
        from tqdm import tqdm
        plies = tqdm(plies, desc="Self-play", leave=False)
    for _ in plies:
        diagnostics = MCTSDiagnostics() if diagnostics_per_ply else None
        pi_moves, root_node = mcts_search(
            gs, model, num_sims=num_mcts_sims, show_progress=show_progress,
            diagnostics=diagnostics, add_root_dirichlet=add_root_dirichlet,
            dir_alpha=dir_alpha, dir_epsilon=dir_epsilon,
            root=root_node, return_root=True, batch_size=16,
        )
        if not pi_moves:
            result = terminal_result(gs)
            if result is None:
                raise RuntimeError("MCTS returned no moves for a nonterminal position.")
            return data, result
        pi = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for mv, probability in pi_moves.items():
            pi[move_to_action(mv)] = probability
        data.append((encode_gamestate(gs), pi, gs.player))
        mv = best_move(pi_moves, root_node) if temperature == 0 else pick_move_from_dist(pi_moves, temperature)
        mover = gs.player
        gs.make_move(mv)
        # Giving check is legal; only the moving side's king must be safe.
        if hasattr(gs, "in_check"):
            assert not gs.in_check(mover), f"Move leaves {mover}'s king in check: {mv}"
        root_node = root_node.children.get(mv) if root_node is not None else None
        result = terminal_result(gs)
        if result is not None:
            return data, result
    return data, 0.0
