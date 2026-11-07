"""Exercise the public executable interfaces, including search cancellation."""
import queue
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chess"))
from scr.chess_engine import GameState


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args, input=None):
        return subprocess.run(
            [sys.executable, "-S", str(ROOT / "chess" / "cli.py"), *args],
            input=input, text=True, capture_output=True, timeout=10, cwd="/tmp",
        )

    def test_perft_without_dependencies_or_working_directory(self):
        result = self.run_cli("perft", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("8902 nodes", result.stdout)

    def test_play_human_and_engine_turn_without_dependencies(self):
        result = self.run_cli("play", "--sims", "10", input="e2e4\nfen\nundo\nquit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Engine plays", result.stdout)
        self.assertNotIn("Invalid move", result.stdout)

    def test_analyze_terminal_position(self):
        result = self.run_cli("analyze", "--fen", "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Checkmate. White wins.", result.stdout)

    def test_invalid_position_is_reported(self):
        result = self.run_cli("analyze", "--fen", "bad fen")
        self.assertEqual(result.returncode, 2)
        self.assertIn("FEN", result.stderr)


class UCIProcessTests(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen(
            [sys.executable, "-S", str(ROOT / "chess" / "cli.py"), "uci"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd="/tmp",
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self.read_output, daemon=True)
        self.reader.start()

    def read_output(self):
        for line in self.process.stdout:
            self.lines.put(line.strip())

    def send(self, text):
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def until(self, prefix, timeout=5):
        deadline = time.monotonic() + timeout
        collected = []
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            collected.append(line)
            if line.startswith(prefix):
                return line, collected
        self.fail(f"Missing {prefix!r}; received {collected}")

    def tearDown(self):
        if self.process.poll() is None:
            self.send("quit")
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=1)
        error = self.process.stderr.read()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
        self.assertEqual(self.process.returncode, 0, error)

    def test_handshake_and_legal_move_after_position_history(self):
        self.send("uci")
        _, lines = self.until("uciok")
        self.assertTrue(any(line.startswith("id name") for line in lines))
        self.send("position startpos moves e2e4 e7e5")
        self.send("go nodes 30")
        best, lines = self.until("bestmove")
        position = GameState()
        position.push_uci("e2e4")
        position.push_uci("e7e5")
        self.assertIsNotNone(position.parse_uci(best.split()[1]))
        self.assertTrue(any("nodes 30 " in line for line in lines))

    def test_isready_during_search_and_stop(self):
        self.send("position startpos")
        self.send("go infinite")
        self.send("isready")
        _, lines = self.until("readyok", timeout=2)
        self.assertFalse(any(line.startswith("bestmove") for line in lines))
        self.send("stop")
        best, _ = self.until("bestmove", timeout=2)
        self.assertIsNotNone(GameState().parse_uci(best.split()[1]))

    def test_movetime_and_checkmate(self):
        self.send("go movetime 40")
        self.until("bestmove", timeout=2)
        self.send("position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
        self.send("go nodes 10")
        best, _ = self.until("bestmove")
        self.assertEqual(best, "bestmove 0000")

    def test_bad_input_preserves_position_and_connection(self):
        self.send("position startpos moves e2e5")
        self.until("info string")
        self.send("isready")
        self.until("readyok")
        self.send("go nodes 2")
        best, _ = self.until("bestmove")
        self.assertIsNotNone(GameState().parse_uci(best.split()[1]))


if __name__ == "__main__":
    unittest.main()
