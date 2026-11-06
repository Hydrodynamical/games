"""A small, asynchronous UCI adapter for the custom rules engine and MCTS."""
from __future__ import annotations

import sys
import threading
import time

from .chess_engine import GameState
from .mcts import MCTSDiagnostics, best_move, mcts_search


class UCIEngine:
    def __init__(self, output=None):
        self.output = output if output is not None else sys.stdout
        self.position = GameState()
        self.simulations = 400
        self.c_puct = 1.5
        self._worker = None
        self._stop = threading.Event()
        self._output_lock = threading.Lock()

    def send(self, message):
        with self._output_lock:
            print(message, file=self.output, flush=True)

    def stop(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join()
            self._worker = None

    def set_position(self, tokens):
        self.stop()
        if not tokens:
            raise ValueError("position requires startpos or fen")
        if tokens[0] == "startpos":
            position, rest = GameState(), tokens[1:]
        elif tokens[0] == "fen" and len(tokens) >= 7:
            position = GameState.from_fen(" ".join(tokens[1:7]))
            rest = tokens[7:]
        else:
            raise ValueError("position requires startpos or six FEN fields")
        if rest:
            if rest[0] != "moves":
                raise ValueError("expected moves after position")
            for move in rest[1:]:
                position.push_uci(move)
        self.position = position

    def set_option(self, tokens):
        if len(tokens) != 4 or tokens[0] != "name" or tokens[2] != "value":
            return
        name, value = tokens[1].lower(), tokens[3]
        if name == "simulations":
            value = int(value)
            if not 1 <= value <= 1_000_000:
                raise ValueError("Simulations must be between 1 and 1000000")
            self.simulations = value
        elif name == "cpuct":
            value = int(value)
            if not 1 <= value <= 1000:
                raise ValueError("CPuct must be between 1 and 1000 (hundredths)")
            self.c_puct = value / 100

    def go(self, tokens):
        self.stop()
        options = {}
        flags = set()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes"}:
                if i + 1 >= len(tokens):
                    raise ValueError(f"missing value for {token}")
                options[token] = int(tokens[i + 1])
                if options[token] < 0:
                    raise ValueError(f"{token} cannot be negative")
                i += 2
            elif token in {"searchmoves", "ponder", "depth", "mate"}:
                raise ValueError(f"{token} is not supported; use nodes or movetime")
            else:
                flags.add(token)
                i += 1

        time_limit = None
        infinite = "infinite" in flags
        if not infinite:
            if "movetime" in options:
                time_limit = options["movetime"] / 1000
            elif self.position.player + "time" in options:
                remaining = options[self.position.player + "time"] / 1000
                increment = options.get(self.position.player + "inc", 0) / 1000
                moves = max(1, options.get("movestogo", 30))
                # Reserve clock time for protocol and move generation overhead.
                time_limit = max(0.0, min(remaining * 0.8, remaining / moves + increment * 0.8) - 0.02)
        simulations = options.get("nodes", 2_000_000_000 if time_limit is not None or infinite else self.simulations)
        position = self.position.fast_copy_for_search()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._search,
            args=(position, simulations, time_limit, infinite, self._stop, self.c_puct),
            daemon=True,
        )
        self._worker.start()

    def _search(self, position, simulations, time_limit, infinite, stop_event, c_puct):
        started = time.monotonic()
        stats = MCTSDiagnostics()
        move = None
        try:
            probabilities, root = mcts_search(
                position, num_sims=simulations, c_puct=c_puct,
                time_limit=time_limit, stop_event=stop_event, diagnostics=stats,
                return_root=True,
            )
            move = best_move(probabilities, root)
        except Exception as exc:
            self.send(f"info string search error: {str(exc).replace(chr(10), ' ')}")
            legal = position.get_all_legal_moves(position.player, as_moves=True)
            move = legal[0] if legal else None
        if infinite:
            stop_event.wait()
        elapsed = max(1, int((time.monotonic() - started) * 1000))
        pv = f" pv {move.uci()}" if move is not None else ""
        self.send(f"info nodes {stats.sims} time {elapsed} nps {stats.sims * 1000 // elapsed}{pv}")
        self.send(f"bestmove {move.uci() if move is not None else '0000'}")

    def command(self, line):
        tokens = line.split()
        if not tokens:
            return True
        command, args = tokens[0], tokens[1:]
        try:
            if command == "uci":
                self.send("id name Hydrodynamical MCTS 0.1")
                self.send("id author Joseph K. Miller")
                self.send("option name Simulations type spin default 400 min 1 max 1000000")
                self.send("option name CPuct type spin default 150 min 1 max 1000")
                self.send("uciok")
            elif command == "isready":
                self.send("readyok")
            elif command == "ucinewgame":
                self.stop()
                self.position = GameState()
            elif command == "position":
                self.set_position(args)
            elif command == "setoption":
                self.set_option(args)
            elif command == "go":
                self.go(args)
            elif command == "stop":
                self.stop()
            elif command == "quit":
                self.stop()
                return False
        except (ValueError, IndexError) as exc:
            self.send(f"info string {exc}")
        return True

    def run(self, source=None):
        try:
            for line in source if source is not None else sys.stdin:
                if not self.command(line):
                    break
        finally:
            self.stop()


def main():
    UCIEngine().run()


if __name__ == "__main__":
    main()
