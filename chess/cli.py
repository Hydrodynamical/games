"""Play, inspect, or serve the chess engine without optional dependencies."""
from __future__ import annotations

import argparse
import time

from scr.chess_engine import GameState
from scr.mcts import MCTSDiagnostics, best_move, mcts_search


def perft(position, depth):
    """Count legal move paths, ignoring game adjudication as perft requires."""
    if depth < 0:
        raise ValueError("depth cannot be negative")
    if depth == 0:
        return 1
    moves = position.get_all_legal_moves(position.player, as_moves=True)
    if depth == 1:
        return len(moves)
    total = 0
    for move in moves:
        position.make_move(move)
        try:
            total += perft(position, depth - 1)
        finally:
            position.undo_last_move()
    return total


def board_text(position):
    rows = []
    for row_index, row in enumerate(position.board):
        symbols = ["." if p == "--" else p[1] if p[0] == "w" else p[1].lower() for p in row]
        rows.append(f"{8 - row_index}  {' '.join(symbols)}")
    return "\n".join(rows) + "\n\n   a b c d e f g h"


def outcome(position):
    if position.is_checkmate():
        return f"Checkmate. {'White' if position.player == 'b' else 'Black'} wins."
    if position.is_stalemate():
        return "Draw by stalemate."
    reason = position.draw_reason()
    if reason:
        return f"Draw by {reason.replace('_', ' ')}."
    return None


def play(position, args):
    print("Hydrodynamical MCTS — enter moves such as e2e4 or a7a8n.")
    print("Commands: moves, undo, fen, help, quit. Uppercase pieces are White.")
    while True:
        print("\n" + board_text(position))
        result = outcome(position)
        if result:
            print(result)
            return
        if position.player != args.you:
            stats = MCTSDiagnostics()
            probabilities, root = mcts_search(position, num_sims=args.sims, time_limit=args.time, diagnostics=stats, return_root=True)
            if not probabilities:
                print("No move available.")
                return
            move = best_move(probabilities, root)
            print(f"Engine plays {move.uci()} ({stats.sims} simulations).")
            position.make_move(move)
            continue
        try:
            text = input(f"{'White' if position.player == 'w' else 'Black'} to move> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text in {"q", "quit", "exit"}:
            return
        if text == "moves":
            print(" ".join(move.uci() for move in position.get_all_legal_moves(position.player, as_moves=True)))
        elif text == "fen":
            print(position.to_fen())
        elif text == "undo":
            # Undo an engine reply and the human move preceding it when available.
            if position.move_log:
                position.undo_last_move()
            if position.player != args.you and position.move_log:
                position.undo_last_move()
        elif text == "help":
            print("Enter source + destination, e.g. e2e4; append q/r/b/n for promotion.")
            print("moves lists legal moves; undo takes back your last turn; fen prints the position; quit exits.")
        else:
            try:
                position.push_uci(text)
            except ValueError as exc:
                print(f"Invalid move: {exc}")


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def seconds(value):
    import math
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("play", "analyze", "perft"):
        command = commands.add_parser(name)
        command.add_argument("--fen", help="start from a six-field FEN position")
        if name == "perft":
            command.add_argument("depth", type=nonnegative)
        else:
            command.add_argument("--sims", type=nonnegative, default=400)
            command.add_argument("--time", type=seconds, default=None, help="maximum search seconds per move")
            if name == "play":
                command.add_argument("--you", choices=("w", "b"), default="w")
    commands.add_parser("uci")
    args = parser.parse_args(argv)
    if args.command == "uci":
        from scr.uci import main as uci_main
        uci_main()
        return
    try:
        position = GameState.from_fen(args.fen) if args.fen else GameState()
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "play":
        play(position, args)
    elif args.command == "perft":
        started = time.monotonic()
        count = perft(position, args.depth)
        print(f"depth {args.depth}: {count} nodes ({time.monotonic() - started:.3f}s)")
    else:
        print(board_text(position))
        result = outcome(position)
        if result:
            print(result)
            return
        stats = MCTSDiagnostics()
        probabilities = mcts_search(position, num_sims=args.sims, time_limit=args.time, diagnostics=stats)
        for move, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f"{move.uci():6} {probability:7.2%}")
        print(f"{stats.sims} simulations")


if __name__ == "__main__":
    main()
