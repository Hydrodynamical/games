from __future__ import annotations
# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500
# main.py

import math
import os
import random
from collections import deque

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


def penalty_schedule(
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
    start: int = 50,
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
    # MPS (Mac) > CUDA (NVIDIA) > CPU device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")


    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    game_idx = 0

    # ----------------------------
    # Replay buffer + minibatching
    # ----------------------------
    # Store (state_tensor, pi_vector, z) on CPU to save GPU memory.
    REPLAY_CAPACITY = 100_000     # number of positions to keep
    BATCH_SIZE = 256
    UPDATES_PER_ITER = 2          # gradient steps per self-play game
    MIN_BUFFER_TO_TRAIN = 2_000   # warm-up before training
    replay = deque(maxlen=REPLAY_CAPACITY)

    for iteration in range(1000):
        print(f"\n=== Iteration {iteration} ===")
        trunc_penalty = penalty_schedule(iteration, p0=0.5, half_life=100.0, floor=0.0)
        draw_penalty = penalty_schedule(iteration, p0=0.5, half_life=100.0, floor=0.0)
        
        print(f"trunc_draw_penalty (annealed): {trunc_penalty:.4f}")
        print(f"draw_penalty (annealed): {draw_penalty:.4f}")

        # Fresh game
        gs = GameState()
        game_idx += 1
        game_data, result = play_self_game(
            gs,
            model,
            num_mcts_sims=mcts_sims_schedule(game_idx, start=50, end=100, warmup_games=100),
            temperature=0.25, # was 0.9
            max_moves=200,
            show_progress=False,
            trunc_draw_penalty=trunc_penalty,  # annealed toward 0 over training
            draw_penalty=draw_penalty,        # annealed toward 0 over training
            diagnostics_per_ply=False,
            add_root_dirichlet=True,
            dir_alpha=0.3,
            dir_epsilon=0.25
        )

        # Convert game to training examples and push into replay buffer
        for state, pi, player in game_data:
            z = result if player == "w" else -result
            # Detach + store on CPU so replay doesn't retain any graphs / GPU tensors
            replay.append((state.detach().cpu(), pi.detach().cpu(), float(z)))

        print(f"Replay size: {len(replay)}")

        # Train only after some buffer warm-up, then do a few minibatch updates
        stats = None
        if len(replay) >= MIN_BUFFER_TO_TRAIN:
            for _ in range(UPDATES_PER_ITER):
                batch_size = min(BATCH_SIZE, len(replay))
                minibatch = random.sample(list(replay), batch_size)
                stats = train_step(
                    model,
                    optimizer,
                    minibatch,
                    device,
                )
            print(stats)
        else:
            print("Warming up replay buffer (skipping training this iter).")

        # Save every 5 iterations
        if iteration % 5 == 0:
            torch.save(model.state_dict(), f"chess/scr/policy_value_net_{iteration}.pt")


if __name__ == "__main__":
    main()
