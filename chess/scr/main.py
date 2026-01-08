from __future__ import annotations
# https://chatgpt.com/share/695c85a9-4ed4-8006-83ab-d4bbc3e7b500
# main.py
import torch

from chess_engine import GameState
from rl_interface import PolicyValueNet
from self_play import play_self_game
from train import train_step


def main():
    #TODO: Figure out how to terminate games. Currently they go on forever.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for iteration in range(1000):
        print(f"\n=== Iteration {iteration} ===")

        # Fresh game
        gs = GameState()

        game_data, result = play_self_game(
            gs,
            model,
            num_mcts_sims=200,
            max_moves=100,
            temperature=1.0,
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

        if iteration % 50 == 0:
            torch.save(model.state_dict(), "policy_value_net.pt")


if __name__ == "__main__":
    main()
