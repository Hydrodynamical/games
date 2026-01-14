# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500
# self_play.py
from typing import List, Tuple
from copy import deepcopy
from tqdm import trange, tqdm
import torch

# from chess_engine import GameState, Move
from rl_interface import (
    encode_gamestate,
    move_to_action,
    NUM_ACTIONS,
)
from mcts import mcts_search, MCTSDiagnostics


def play_self_game(
    gs,
    model,
    num_mcts_sims: int = 30,
    temperature: float = 2,
    max_moves: int = 200,
    show_progress: bool = False,
    trunc_draw_penalty: float = 0.2,  # penalty for truncated games, positive
    draw_penalty: float = 0.02,        # penalty for non-truncation draws (3-fold/50-move), positive
    diagnostics_per_ply: bool = False,
    add_root_dirichlet: bool = True,
    dir_alpha: float = 0.3,
    dir_epsilon: float = 0.25
):
    """
    Play one self-play game using MCTS.

    Returns:
        data: list of (state_tensor, pi_vector, player)
        result: final game result from White's perspective (win/loss as ±1; draws may return a small ±penalty)
    """
    data = []
    gs = deepcopy(gs)

    # NEW: keep MCTS tree root across plies (tree reuse)
    root_node = None

    # Play until termination or max_moves
    for ply in trange(max_moves,desc="Self-Play Game", leave=False):
        # MCTS → improved policy
        if diagnostics_per_ply:
            per_move = MCTSDiagnostics()
        else:
            per_move = None
        out = mcts_search(
            gs, 
            model, 
            num_sims=num_mcts_sims, 
            show_progress=show_progress,
            diagnostics = per_move,
            add_root_dirichlet=add_root_dirichlet,
            dir_alpha=dir_alpha,
            dir_epsilon=dir_epsilon,
            root=root_node,
            return_root=True,
            batch_size=16,
            )
        pi_moves, root_node = out
        
        # Build full π vector in 4096 action space
        pi = torch.zeros(NUM_ACTIONS)
        for mv, p in pi_moves.items():
            pi[move_to_action(mv)] = p

        # Store training triple
        state_tensor = encode_gamestate(gs)
        data.append((state_tensor, pi, gs.player))

        # Sample move (temperature-controlled)
        moves, probs = zip(*pi_moves.items())
        if temperature == 0:
            mv = moves[max(range(len(probs)), key=lambda i: probs[i])]
        else:
            idx = torch.multinomial(torch.tensor(probs), 1).item()
            mv = moves[idx]

        # Play the move (this should also switch gs.player internally)
        gs.make_move(mv)

        # NEW: advance the root to the played move (reuse subtree next ply)
        if root_node is not None and mv in root_node.children:
            root_node = root_node.children[mv]
        else:
            root_node = None

        if hasattr(gs, "in_check"):
            # ensure current player's king is not left in check (illegal position)
            assert not gs.in_check(gs.player), f"Illegal position: {gs.player} is in check after move {mv}"

        # After make_move, check draw rules
        if hasattr(gs, "is_draw") and gs.is_draw():
            print("\n=== DRAW ===")
            if hasattr(gs, "draw_reason"):
                print(f"Reason: {gs.draw_reason()}")
            print(f"Last move: {mv}")
            # Penalize the player who *just moved* for causing the draw
            # (3-fold repetition / 50-move, etc.), rather than always penalizing White.
            # Result is from White's perspective:
            #   - if White caused the draw => negative
            #   - if Black caused the draw => positive (penalizes Black)
            mover = gs.opponent_color() if hasattr(gs, "opponent_color") else ("b" if gs.player == "w" else "w")
            return data, (-float(draw_penalty) if mover == "w" else +float(draw_penalty))

        # After make_move, gs.player is the opponent of the player who just moved.
        # If gs.player is checkmated, the winner is the other side.
        if gs.is_checkmate():
            winner = gs.opponent_color() if hasattr(gs, "opponent_color") else ("b" if gs.player == "w" else "w")
            print("\n=== CHECKMATE ===")
            print(f"Winner: {winner}")
            print(f"Winning move: {mv}")
            return data, (+1 if winner == "w" else -1)

        if gs.is_stalemate():
            print("\n=== STALEMATE ===")
            print(f"Last move: {mv}")
            # Stalemate is a draw, but we penalize the player who delivered it.
            mover = gs.opponent_color() if hasattr(gs, "opponent_color") else ("b" if gs.player == "w" else "w")
            return data, (-float(draw_penalty) if mover == "w" else +float(draw_penalty))

    # If we hit max_moves without termination, treat as draw by truncation
    print("\n=== TRUNCATION DRAW ===")
    print(f"Reached max_moves={max_moves} without checkmate/stalemate.")
    # Penalize the player who made the *last move* that led into the truncation.
    mover = gs.opponent_color() if hasattr(gs, "opponent_color") else ("b" if gs.player == "w" else "w")
    return data, (-float(trunc_draw_penalty) if mover == "w" else +float(trunc_draw_penalty))