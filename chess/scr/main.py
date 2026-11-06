"""Bounded, configurable neural MCTS self-play training.

Example: python chess/scr/main.py --iterations 2 --sims 8 --max-moves 8
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import random

import torch

try:
    from .chess_engine import GameState
    from .rl_interface import PolicyValueNet, load_model, save_model, select_device
    from .self_play import play_self_game
    from .train import train_step
except ImportError:
    from chess_engine import GameState
    from rl_interface import PolicyValueNet, load_model, save_model, select_device
    from self_play import play_self_game
    from train import train_step


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=positive_int, default=1000, help="Self-play games to generate")
    parser.add_argument("--sims", type=positive_int, default=100, help="MCTS simulations per ply")
    parser.add_argument("--max-moves", type=positive_int, default=200, help="Maximum plies per game")
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--min-buffer", type=positive_int, default=2000, help="Positions before the first training step")
    parser.add_argument("--replay-capacity", type=positive_int, default=20000)
    parser.add_argument("--updates", type=positive_int, default=2, help="Training steps per game after warmup")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--save-every", type=positive_int, default=5)
    parser.add_argument("--resume", type=Path, help="Load network weights; optimizer/replay start fresh")
    parser.add_argument("--hidden", type=positive_int, default=128, help="Convolution channels for a new model")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.min_buffer > args.replay_capacity:
        parser.error("--min-buffer must not exceed --replay-capacity")
    from math import isfinite
    if not isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else select_device()
    model = load_model(args.resume, device) if args.resume else PolicyValueNet(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay = deque(maxlen=args.replay_capacity)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training on {device}; {args.iterations} games, {args.sims} simulations/ply")
    for iteration in range(1, args.iterations + 1):
        game_data, result = play_self_game(
            GameState(), model, num_mcts_sims=args.sims, temperature=args.temperature,
            max_moves=args.max_moves, show_progress=args.progress,
        )
        for state, policy, player in game_data:
            value = result if player == "w" else -result
            replay.append((state.detach().cpu(), policy.detach().cpu(), float(value)))
        print(f"Game {iteration}: {len(game_data)} plies, result={result:+.0f}, replay={len(replay)}")
        if len(replay) >= args.min_buffer:
            for _ in range(args.updates):
                batch = random.sample(list(replay), min(args.batch_size, len(replay)))
                stats = train_step(model, optimizer, batch, device)
            print(f"Loss: {stats['loss']:.5f} (policy={stats['policy_loss']:.5f}, value={stats['value_loss']:.5f})")
        else:
            print("Warming up replay buffer")
        if iteration % args.save_every == 0 or iteration == args.iterations:
            path = args.checkpoint_dir / f"policy_value_net_{iteration}.pt"
            save_model(model, path)
            print(f"Saved {path}")
    return model


if __name__ == "__main__":
    main()
