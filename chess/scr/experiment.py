"""Durable, dependency-free records for inspecting a training experiment."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RunLogger:
    """Append complete JSON lines and atomically refresh a compact summary."""

    def __init__(self, directory, config, *, torch_version, source_root):
        self.directory = Path(directory).resolve()
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError(f"Run directory is not empty: {self.directory}; choose a new directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        source_root = Path(source_root).resolve()
        self.games = []
        self.update_count = 0
        self.last_update = None
        self.checkpoints = []
        self.started_at = utc_now()
        hashes = {}
        # Capture the actual source used even when the working tree is dirty.
        sources = list((source_root / "chess" / "scr").glob("*.py"))
        sources += list((source_root / "chess" / "scr").glob("*.html"))
        sources += list((source_root / "requirements").glob("*.txt"))
        for path in sorted(sources):
            relative = path.relative_to(source_root)
            content = path.read_bytes()
            destination = self.directory / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()

        def git(*args):
            try:
                return subprocess.check_output(["git", *args], cwd=source_root, text=True,
                                               stderr=subprocess.DEVNULL, timeout=5).strip()
            except (OSError, subprocess.SubprocessError):
                return None

        metadata = {
            "schema_version": 1, "started_at": self.started_at, "config": config,
            "python_version": sys.version, "torch_version": str(torch_version),
            "platform": platform.system(), "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"),
            "git_dirty": bool(git("status", "--porcelain")), "source_sha256": hashes,
            "value_perspective": "player to move",
            "truncation_policy": "move_limit has outcome '*' but training target 0",
            "resume_policy": "weights only; optimizer and replay start fresh",
        }
        write_json(self.directory / "config.json", metadata)
        self.refresh()

    def append(self, filename, value):
        line = json.dumps(value, allow_nan=False, separators=(",", ":"))
        with (self.directory / filename).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def game(self, record):
        self.append("games.jsonl", record)
        # Keep only aggregate inputs in memory; traces stay in the append log.
        self.games.append({key: record[key] for key in
                           ("game", "termination", "truncated", "outcome", "result", "plies", "duration_seconds")})

    def update(self, record):
        self.append("updates.jsonl", record)
        self.update_count += 1
        self.last_update = record

    def probes(self, record):
        self.append("probes.jsonl", record)

    def checkpoint(self, path, game, step):
        self.checkpoints.append({"path": str(Path(path).resolve()), "game": game, "step": step})

    def refresh(self, status="running", error=None, replay=None):
        count = len(self.games)
        endings = Counter(game["termination"] for game in self.games)
        cutoffs = sum(game["truncated"] for game in self.games)
        actual_draws = sum(game["outcome"] == "1/2-1/2" for game in self.games)
        positions = sum(game["plies"] for game in self.games)
        cutoff_positions = sum(game["plies"] for game in self.games if game["truncated"])
        summary = {
            "schema_version": 1, "status": status, "started_at": self.started_at,
            "updated_at": utc_now(), "games_completed": count, "termination_counts": dict(endings),
            "actual_draws": actual_draws, "decisive_games": endings["checkmate"],
            "move_limit_games": cutoffs, "move_limit_fraction": cutoffs / count if count else 0.0,
            "positions_generated": positions, "cutoff_target_positions": cutoff_positions,
            "cutoff_target_fraction": cutoff_positions / positions if positions else 0.0,
            "mean_plies": positions / count if count else 0.0,
            "self_play_seconds": sum(game["duration_seconds"] for game in self.games),
            "optimizer_steps": self.update_count, "last_update": self.last_update,
            "checkpoints": self.checkpoints, "replay": replay, "error": error,
        }
        write_json(self.directory / "summary.json", summary)
        return summary
