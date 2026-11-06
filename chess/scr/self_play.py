"""Generate policy targets by MCTS self-play and neutral draw value targets."""
from __future__ import annotations

from copy import deepcopy
import math
import time
import warnings

import torch

try:
    from .rl_interface import encode_gamestate, move_to_action, NUM_ACTIONS, pick_move_from_dist, critic_value
    from .mcts import mcts_search, MCTSDiagnostics, best_move
except ImportError:
    from rl_interface import encode_gamestate, move_to_action, NUM_ACTIONS, pick_move_from_dist, critic_value
    from mcts import mcts_search, MCTSDiagnostics, best_move


def terminal_details(gs):
    """Return (termination, White's result), or None while playing."""
    if gs.is_checkmate():
        return "checkmate", -1.0 if gs.player == "w" else 1.0
    if gs.is_stalemate():
        return "stalemate", 0.0
    reason = gs.draw_reason()
    if reason is not None:
        return reason, 0.0
    return None


def terminal_result(gs):
    """White's result, or None while playing. Checkmate takes precedence."""
    details = terminal_details(gs)
    return details[1] if details is not None else None


def _entropy(probabilities):
    """Shannon entropy in nats, with zero-probability terms omitted."""
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def _trace_position(gs, model, pi_moves, root, move, ply, diagnostics, search_seconds):
    """Describe the policy/search decision before applying the chosen move.

    Priors are the clean, legal-move distribution before root exploration
    noise. Search probabilities include all visits retained in a reused root;
    ``sims`` counts only the additional simulations performed this ply.
    """
    priors = root._base_priors
    value = float(critic_value(model, gs)) if model is not None else None
    if value is not None and not math.isfinite(value):
        value = None
    top_moves = sorted(
        pi_moves, key=lambda mv: (pi_moves[mv], root.edges[mv].q, root.edges[mv].prior), reverse=True,
    )[:5]
    return {
        "ply": ply,
        "fen": gs.to_fen(),
        "player": gs.player,
        "move": move.uci(),
        "value_prediction": value,
        "legal_moves": len(pi_moves),
        "policy_entropy": _entropy(priors.values()),
        "search_entropy": _entropy(pi_moves.values()),
        "policy_search_kl": max(0.0, sum(
            p * math.log(p / max(priors[mv], 1e-12)) for mv, p in pi_moves.items() if p > 0
        )),
        "policy_top_move": max(priors, key=priors.get).uci(),
        "search_top_move": best_move(pi_moves, root).uci(),
        "selected_policy_probability": priors[move],
        "selected_search_probability": pi_moves[move],
        "sims": diagnostics.sims,
        "search_seconds": search_seconds,
        "top_moves": [
            {"move": mv.uci(), "prior": priors[mv], "search_probability": pi_moves[mv],
             "visits": root.edges[mv].n, "q": root.edges[mv].q}
            for mv in top_moves
        ],
    }


def play_self_game(
    gs, model, num_mcts_sims: int = 30, temperature: float = 1.0,
    max_moves: int = 200, show_progress: bool = False,
    trunc_draw_penalty: float = 0.0, draw_penalty: float = 0.0,
    diagnostics_per_ply: bool = False, add_root_dirichlet: bool = True,
    dir_alpha: float = 0.3, dir_epsilon: float = 0.25,
    return_info: bool = False, record_trace: bool = False,
):
    """Return (state, visit-policy, mover) samples and White's result.

    Draws and games stopped at max_moves are scored zero. The legacy penalty
    arguments are accepted but ignored: last-mover penalties create incorrect
    zero-sum training labels. Temperature controls played moves; training policy
    targets remain the normalized MCTS visit counts.

    ``return_info=True`` adds a JSON-serializable third return value describing
    why the game ended and its UCI move history. A move-limit cutoff has outcome
    "*" rather than a chess draw, although its existing training label remains
    zero. ``record_trace=True`` also records network/search decisions per ply;
    this adds one network evaluation per position for the raw value prediction.
    """
    if max_moves < 0 or num_mcts_sims < 0:
        raise ValueError("Move and simulation budgets must be nonnegative.")
    # Validate temperature even for an initially terminal game.
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    if trunc_draw_penalty or draw_penalty:
        warnings.warn("Draw penalties are ignored; chess draws have value zero.", DeprecationWarning, stacklevel=2)
    started = time.monotonic()
    gs = deepcopy(gs)
    data, root_node = [], None
    initial_fen = gs.to_fen() if return_info else None
    moves, trace = [], []

    def finish(termination, result):
        if not return_info:
            return data, result
        truncated = termination == "move_limit"
        winner = ("w" if result > 0 else "b") if result else None
        info = {
            "termination": termination,
            "truncated": truncated,
            "outcome": "*" if truncated else "1-0" if result > 0 else "0-1" if result < 0 else "1/2-1/2",
            "winner": winner,
            "result": result,
            "plies": len(moves),
            "initial_fen": initial_fen,
            "final_fen": gs.to_fen(),
            "moves": moves,
            "duration_seconds": time.monotonic() - started,
            "trace": trace,
        }
        return data, result, info

    ending = terminal_details(gs)
    if ending is not None:
        return finish(*ending)
    plies = range(max_moves)
    if show_progress:
        from tqdm import tqdm
        plies = tqdm(plies, desc="Self-play", leave=False)
    for ply in plies:
        diagnostics = MCTSDiagnostics() if diagnostics_per_ply or record_trace else None
        search_started = time.monotonic()
        pi_moves, root_node = mcts_search(
            gs, model, num_sims=num_mcts_sims, show_progress=show_progress,
            diagnostics=diagnostics, add_root_dirichlet=add_root_dirichlet,
            dir_alpha=dir_alpha, dir_epsilon=dir_epsilon,
            root=root_node, return_root=True, batch_size=16,
        )
        search_seconds = time.monotonic() - search_started
        if not pi_moves:
            ending = terminal_details(gs)
            if ending is None:
                raise RuntimeError("MCTS returned no moves for a nonterminal position.")
            return finish(*ending)
        pi = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for mv, probability in pi_moves.items():
            pi[move_to_action(mv)] = probability
        data.append((encode_gamestate(gs), pi, gs.player))
        mv = best_move(pi_moves, root_node) if temperature == 0 else pick_move_from_dist(pi_moves, temperature)
        if record_trace:
            trace.append(_trace_position(gs, model, pi_moves, root_node, mv, ply + 1,
                                         diagnostics, search_seconds))
        mover = gs.player
        gs.make_move(mv)
        moves.append(mv.uci())
        # Giving check is legal; only the moving side's king must be safe.
        if hasattr(gs, "in_check"):
            assert not gs.in_check(mover), f"Move leaves {mover}'s king in check: {mv}"
        root_node = root_node.children.get(mv) if root_node is not None else None
        ending = terminal_details(gs)
        if ending is not None:
            return finish(*ending)
    return finish("move_limit", 0.0)
