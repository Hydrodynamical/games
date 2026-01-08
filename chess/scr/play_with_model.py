# play_with_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import argparse
import torch

# Your project modules
from mcts import mcts_search
from rl_interface import policy_distribution, critic_value, PolicyValueNet

# If your GameState lives elsewhere, update this import accordingly:
from chess_engine import GameState


def move_to_str(mv: Any) -> str:
    """
    Best-effort pretty printer for your Move objects.
    Falls back to repr(mv) if attributes aren't present.
    """
    # common pattern: mv.start, mv.end are tuples like (r,c)
    if hasattr(mv, "start") and hasattr(mv, "end"):
        s = getattr(mv, "start")
        e = getattr(mv, "end")
        pm = getattr(mv, "piece_moved", "")
        pc = getattr(mv, "piece_captured", "")
        extra = []
        if getattr(mv, "is_castle", False):
            extra.append("castle")
        if getattr(mv, "is_en_passant", False):
            extra.append("ep")
        tag = f" [{' '.join(extra)}]" if extra else ""
        cap = f" x{pc}" if pc not in ("--", "", None) else ""
        return f"{pm}:{s}->{e}{cap}{tag}"
    if hasattr(mv, "__str__"):
        try:
            return str(mv)
        except Exception:
            pass
    return repr(mv)


def board_to_ascii(gs: Any) -> str:
    """
    Best-effort board display.
    Uses gs.board if present and it looks like an 8x8 array of piece codes.
    """
    b = getattr(gs, "board", None)
    if b is None:
        return "<no gs.board to display>"
    lines = []
    for r in range(len(b)):
        row = b[r]
        lines.append(" ".join(f"{x:>2}" for x in row))
    return "\n".join(lines)


def top_k(d: Dict[Any, float], k: int = 10) -> List[Tuple[Any, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]


def pick_move_from_dist(dist: Dict[Any, float], temperature: float) -> Any:
    """
    Sample a move from a dict {move: prob}.
    Implements a real temperature transform:
      - T=0 => argmax
      - else => p_i^(1/T) renormalized
    """
    moves = list(dist.keys())
    probs = torch.tensor([float(dist[m]) for m in moves], dtype=torch.float32)

    if len(moves) == 0:
        return None

    if temperature == 0.0:
        return moves[int(torch.argmax(probs).item())]

    # avoid zeros / NaNs
    probs = torch.clamp(probs, min=1e-12)
    probs = probs ** (1.0 / float(temperature))
    probs = probs / probs.sum()

    idx = int(torch.multinomial(probs, 1).item())
    return moves[idx]


def inspect_position(gs: GameState, model: torch.nn.Module, k: int = 10) -> None:
    """
    Print value + top-k policy priors for current position.
    """
    v = float(critic_value(model, gs))
    priors = policy_distribution(model, gs)

    print("\n=== Position inspection ===")
    print(f"Player to move: {getattr(gs, 'player', '?')}")
    print(f"Value (from side-to-move perspective): {v:+.3f}")

    if not priors:
        print("No legal moves / empty priors.")
        return

    # sanity: sum of priors
    s = sum(float(p) for p in priors.values())
    print(f"Priors: {len(priors)} legal moves, sum={s:.6f}")

    print(f"\nTop {k} priors:")
    for mv, p in top_k(priors, k=k):
        print(f"  {p:8.5f}  {move_to_str(mv)}")
    print()


def interactive_game(
    model_path: str,
    you_play: str,
    use_mcts: bool,
    num_sims: int,
    temperature: float,
    show_priors_each_turn: bool,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    state_dict = torch.load(model_path, map_location=device)
    model = PolicyValueNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # model = torch.load(model_path, map_location=device)
    # model.eval()

    gs = GameState()  # assumes default constructor creates initial chess position

    print("Loaded model:", model_path)
    print("Device:", device)
    print("You play as:", you_play)
    print("Use MCTS:", use_mcts, f"(num_sims={num_sims})")
    print("Temperature:", temperature)
    print()

    turn = 0
    while True:
        turn += 1
        print("\n" + "=" * 60)
        print(f"Turn {turn} | player to move: {getattr(gs, 'player', '?')}")
        print(board_to_ascii(gs))

        # Terminal checks (best-effort)
        if hasattr(gs, "is_checkmate") and gs.is_checkmate():
            # side to move is checkmated => other side won
            winner = "w" if getattr(gs, "player", "w") == "b" else "b"
            print(f"\nCHECKMATE. Winner: {winner}")
            return
        if hasattr(gs, "is_stalemate") and gs.is_stalemate():
            print("\nSTALEMATE. Draw.")
            return

        # Generate legal moves
        legal = gs.get_all_legal_moves(gs.player, as_moves=True)
        if not legal:
            print("\nNo legal moves. (Terminal) Draw-ish.")
            return

        if show_priors_each_turn:
            inspect_position(gs, model, k=10)

        # Decide whether human or model moves
        player = getattr(gs, "player", "w")
        human_turn = (player == you_play)

        if human_turn:
            print("\nYour move. Legal moves:")
            for i, mv in enumerate(legal):
                print(f"  [{i:02d}] {move_to_str(mv)}")

            while True:
                raw = input("Enter move index (or 'q' to quit): ").strip()
                if raw.lower() in ("q", "quit", "exit"):
                    print("Bye.")
                    return
                try:
                    idx = int(raw)
                    if 0 <= idx < len(legal):
                        mv = legal[idx]
                        break
                except ValueError:
                    pass
                print("Invalid input. Try again.")
        else:
            print("\nModel thinking...")

            if use_mcts:
                pi_moves = mcts_search(gs, model, num_sims=num_sims, show_progress=True)
                mv = pick_move_from_dist(pi_moves, temperature=temperature)

                # show top MCTS choices
                print("\nTop MCTS moves:")
                for m, p in top_k(pi_moves, k=10):
                    print(f"  {p:8.5f}  {move_to_str(m)}")
            else:
                priors = policy_distribution(model, gs)
                mv = pick_move_from_dist(priors, temperature=temperature)

                print("\nTop prior moves:")
                for m, p in top_k(priors, k=10):
                    print(f"  {p:8.5f}  {move_to_str(m)}")

            if mv is None:
                print("Model returned no move. Ending.")
                return

            print("\nModel plays:", move_to_str(mv))

        # Apply move
        gs.make_move(mv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="policy_value_net.pt")
    ap.add_argument("--you", type=str, choices=["w", "b"], default="w")
    ap.add_argument("--no-mcts", action="store_true", help="Use raw policy priors instead of MCTS")
    ap.add_argument("--sims", type=int, default=200, help="MCTS simulations per move (if using MCTS)")
    ap.add_argument("--temp", type=float, default=1.0, help="Sampling temperature (0 = greedy)")
    ap.add_argument("--show-priors", action="store_true", help="Print value + top priors each turn")
    args = ap.parse_args()

    interactive_game(
        model_path=args.model,
        you_play=args.you,
        use_mcts=not args.no_mcts,
        num_sims=args.sims,
        temperature=args.temp,
        show_priors_each_turn=args.show_priors,
    )


if __name__ == "__main__":
    main()
