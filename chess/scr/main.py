from __future__ import annotations
# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500
# main.py

import math
import os
import random

import numpy as np
import torch
print("torch version:", torch.__version__)

from chess_engine import GameState
print(f"chess_engine.py being used:", GameState.__module__)

from rl_interface import PolicyValueNet
print("rl_interface.py being used:", PolicyValueNet.__module__)

import mcts
print("mcts.py being used:", mcts.__file__)

from self_play import play_self_game
print("self_play.py being used:", play_self_game.__module__)

from train import train_step
print("train.py being used:", train_step.__module__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # More deterministic behavior (may reduce performance)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Best set before Python starts, but harmless to set here too.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def trunc_penalty_schedule(
    iteration: int,
    p0: float = 0.2,
    half_life: float = 10.0,
    floor: float = 0.0,
) -> float:
    """
    Exponentially decay truncation penalty from p0 toward floor.
    half_life = number of iterations to halve the penalty.
    """
    if half_life <= 0:
        return max(floor, p0)
    p = p0 * math.pow(0.5, (iteration / half_life))
    return max(floor, float(p))

def mcts_sims_schedule(
    game_idx: int,
    start: int = 10,
    end: int = 200,
    warmup_games: int = 100,
) -> int:
    """
    Linearly anneal max_moves from start to end over warmup_games.
    """
    if game_idx >= warmup_games:
        return end
    frac = game_idx / float(warmup_games)
    return int(round(start + frac * (end - start)))

def main():
    """This function runs self-play training iterations."""
    SEED = 1
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    game_idx = 0

    for iteration in range(1000):
        print(f"\n=== Iteration {iteration} ===")
        trunc_penalty = trunc_penalty_schedule(iteration, p0=0.2, half_life=10.0, floor=0.0)
        print(f"trunc_draw_penalty (annealed): {trunc_penalty:.4f}")

        # Fresh game
        gs = GameState()
        game_idx += 1
        game_data, result = play_self_game(
            gs,
            model,
            num_mcts_sims=mcts_sims_schedule(game_idx, start=50, end=300, warmup_games=100),
            temperature=0.25, # was 0.9
            max_moves=200,
            trunc_draw_penalty=trunc_penalty,  # annealed toward 0 over training
            diagnostics_per_ply=True,
            add_root_dirichlet=False,
            dir_alpha=0.3,
            dir_epsilon=0.25
        )

        # Assign z values
        training_batch = []
        for state, pi, player in game_data:
            z = result if player == "w" else -result
            training_batch.append((state, pi, z))

        stats = train_step(
            model,
            optimizer,
            training_batch,
            device,
        )

        print(stats)

        # Save every iteration (depth is deep, takes a long time to generate one iteration)
        torch.save(model.state_dict(), f"chess/scr/policy_value_net_{iteration}.pt")


if __name__ == "__main__":
    main()
