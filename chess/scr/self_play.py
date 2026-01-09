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
    trunc_draw_penalty: float = 0.2, # penalty for truncated games, positive
    diagnostics_per_ply: bool = False,
    add_root_dirichlet: bool = True,
    dir_alpha: float = 0.3,
    dir_epsilon: float = 0.25
):
    """
    Play one self-play game using MCTS.

    Returns:
        data: list of (state_tensor, pi_vector, player)
        result: final game result in {-1, 0, +1} from White's perspective
    """
    data = []
    gs = deepcopy(gs)

    # Play until termination or max_moves
    for ply in trange(max_moves,desc="Self-Play Game", leave=False):
        # MCTS → improved policy
        if diagnostics_per_ply:
            per_move = MCTSDiagnostics()
        else:
            per_move = None
        pi_moves = mcts_search(
            gs, 
            model, 
            num_sims=num_mcts_sims, 
            show_progress=True, 
            diagnostics = per_move,
            add_root_dirichlet=add_root_dirichlet,
            dir_alpha=dir_alpha,
            dir_epsilon=dir_epsilon
            )
        
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

        if hasattr(gs, "in_check"):
            # ensure current player's king is not left in check (illegal position)
            assert not gs.in_check(gs.player), f"Illegal position: {gs.player} is in check after move {mv}"


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
            return data, 0

    # If we hit max_moves without termination, treat as draw by truncation
    print("\n=== TRUNCATION DRAW ===")
    print(f"Reached max_moves={max_moves} without checkmate/stalemate.")
    return data, -trunc_draw_penalty