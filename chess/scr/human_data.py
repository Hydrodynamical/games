"""Validate human PGNs and build bounded-memory supervised minibatches.

Run ``PYTHONPATH=chess python -m scr.human_data --input games.jsonl
--output-dir prepared``. PGNs are parsed by python-chess, then every move is
replayed by the custom rules engine before encoding. A game's positions always
stay together; chronological date groups define the train/validation/test split.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

import chess
import chess.pgn
import numpy as np
import torch

try:
    from .chess_engine import GameState
    from .rl_interface import encode_gamestate, move_to_action, NUM_ACTIONS
except ImportError:
    from chess_engine import GameState
    from rl_interface import encode_gamestate, move_to_action, NUM_ACTIONS


SPLITS = ("train", "validation", "test")
RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _username(value):
    return str(value.get("username", "") if isinstance(value, dict) else value).casefold()


def parse_record(record):
    """Validate one downloaded PGN and return its portable full-history record.

    A recorded result is accepted for resignation, timeout, and agreed draws;
    the final board need not itself be terminal. Incomplete/malformed PGNs and
    inconsistent player/result metadata are rejected rather than shortened.
    """
    if record.get("rules", "chess") != "chess":
        raise ValueError("unsupported_variant")
    result = record.get("result")
    if result not in RESULTS:
        raise ValueError("unknown_result")
    pgn = record.get("pgn")
    if not isinstance(pgn, str) or not pgn.strip():
        raise ValueError("missing_pgn")
    trailing = r"(?:\s|\{[^}]*\}|;[^\n]*(?:\n|$))*$"
    if re.search(re.escape(result) + trailing, pgn, re.DOTALL) is None:
        raise ValueError("missing_or_inconsistent_movetext_result")
    stream = io.StringIO(pgn)
    try:
        game = chess.pgn.read_game(stream)
    except (ValueError, IndexError) as exc:
        raise ValueError("invalid_pgn") from exc
    if game is None or game.errors:
        raise ValueError("invalid_pgn")
    if chess.pgn.read_game(stream) is not None:
        raise ValueError("multiple_games_in_record")
    if game.headers.get("Variant", "Standard").casefold() not in {"standard", "chess"}:
        raise ValueError("unsupported_variant")
    if game.headers.get("Result") != result:
        raise ValueError("inconsistent_result")
    white, black = _username(record.get("white", "")), _username(record.get("black", ""))
    if not white or not black or (_username(game.headers.get("White", "")),
                                 _username(game.headers.get("Black", ""))) != (white, black):
        raise ValueError("inconsistent_players")
    target_color = record.get("target_color")
    if target_color not in {"w", "b"}:
        raise ValueError("missing_target_color")
    end_time = record.get("end_time")
    if not isinstance(end_time, (int, float)) or isinstance(end_time, bool) or not np.isfinite(end_time) or end_time < 0:
        raise ValueError("invalid_end_time")
    if not record.get("game_id"):
        raise ValueError("missing_game_id")
    board = game.board()
    if board.chess960 or not board.is_valid():
        raise ValueError("invalid_initial_position")
    initial_fen = board.fen(en_passant="fen")
    moves = []
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            raise ValueError("illegal_pgn_move")
        moves.append(move.uci())
        board.push(move)
    if not moves:
        raise ValueError("empty_game")
    # Automatic terminal results must agree. Claimable draws may legitimately
    # be declined, so python-chess's default claim_draw=False is intentional.
    outcome = board.outcome()
    if outcome is not None and outcome.result() != result:
        raise ValueError("inconsistent_terminal_result")
    normalized = {key: record.get(key) for key in
                  ("game_id", "url", "end_time", "termination", "time_control", "time_class", "rated", "archive_url")}
    normalized.update(white=white, black=black, target_color=target_color,
                      result=result, initial_fen=initial_fen, moves=moves,
                      final_fen=board.fen(en_passant="fen"), plies=len(moves))
    # Distinguish separate games that happen to share their opening/movetext.
    # Exact move/players/time duplicates under a different URL are still one game.
    identity = [white, black, end_time, game.headers.get("UTCDate", game.headers.get("Date")),
                game.headers.get("UTCTime", game.headers.get("StartTime")),
                initial_fen, moves, result]
    normalized["game_fingerprint"] = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
    return normalized


def assign_splits(records):
    """Assign date-grouped chronological splits nearest to 80/10/10.

    At least three distinct dates are required. This intentionally avoids
    presenting random positions from the same session as a held-out test.
    """
    records.sort(key=lambda row: (row["end_time"], str(row["game_id"])))
    groups = []
    for index, record in enumerate(records):
        day = datetime.fromtimestamp(record["end_time"], timezone.utc).date().isoformat()
        if not groups or groups[-1][0] != day:
            groups.append((day, index))
    if len(groups) < 3:
        raise ValueError("At least three UTC dates are required for chronological train/validation/test splits")
    count = len(records)
    train_group = min(range(1, len(groups) - 1), key=lambda i: abs(groups[i][1] - .8 * count))
    validation_group = min(range(train_group + 1, len(groups)), key=lambda i: abs(groups[i][1] - .9 * count))
    train_end, validation_end = groups[train_group][1], groups[validation_group][1]
    for index, record in enumerate(records):
        record["game_index"] = index
        record["split"] = "train" if index < train_end else "validation" if index < validation_end else "test"
    return {"strategy": "chronological_whole_utc_date_groups", "target_fractions": [.8, .1, .1],
            "validation_start_utc_date": groups[train_group][0], "test_start_utc_date": groups[validation_group][0]}


def encode_record(record):
    """Return compact arrays for one fully validated game, or fail atomically."""
    gs = GameState.from_fen(record["initial_fen"])
    oracle = chess.Board(record["initial_fen"])
    states, actions, values, policy_mask = [], [], [], []
    white_value = RESULTS[record["result"]]
    for ply, uci in enumerate(record["moves"]):
        if gs.to_fen() != oracle.fen(en_passant="fen"):
            raise ValueError(f"Custom engine differs from PGN at game {record['game_id']}, ply {ply}")
        # Encoding before pushing preserves the actual repeated-position count.
        encoded = encode_gamestate(gs).numpy()
        encoded[18] *= 100
        encoded[19] *= 3
        states.append(np.rint(encoded).astype(np.uint8))
        values.append(white_value if gs.player == "w" else -white_value)
        policy_mask.append(gs.player == record["target_color"])
        move = gs.push_uci(uci)
        actions.append(move_to_action(move))
        oracle.push_uci(uci)
    if gs.to_fen() != oracle.fen(en_passant="fen"):
        raise ValueError(f"Custom engine differs from PGN after game {record['game_id']}")
    return {"states": np.stack(states), "actions": np.asarray(actions, dtype=np.int16),
            "values": np.asarray(values, dtype=np.int8), "policy_mask": np.asarray(policy_mask, dtype=np.bool_),
            "game_indices": np.full(len(actions), record["game_index"], dtype=np.int32),
            "plies": np.arange(len(actions), dtype=np.int32)}


def prepare_dataset(input_path, output_dir, shard_size=8192):
    """Prepare compact shards and provenance; existing output is never overwritten."""
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty prepared directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "status": "preparing", "source_path": str(input_path),
                "source_sha256": _sha256(input_path), "shard_size": shard_size,
                "encoding": "20 uint8 planes; divide plane 18 by 100 and plane 19 by 3",
                "policy_supervision": "target player's own turns only",
                "value_supervision": "observed PGN result, both sides, player-to-move perspective",
                "value_limitations": "Human outcomes include resignation/timeouts and are not perfect-play evaluations",
                "input_records": 0, "accepted_games": 0, "rejected_counts": {}, "shards": [], "splits": {}}
    _write_json(output_dir / "manifest.json", manifest)
    try:
        records, rejected, ids, fingerprints = [], Counter(), set(), set()
        with input_path.open() as stream, (output_dir / "rejected.jsonl").open("w") as rejected_stream:
            for line_index, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                manifest["input_records"] += 1
                record = {}
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record_not_object")
                    normalized = parse_record(record)
                    if str(normalized["game_id"]) in ids:
                        raise ValueError("duplicate_game_id")
                    if normalized["game_fingerprint"] in fingerprints:
                        raise ValueError("duplicate_game_fingerprint")
                    ids.add(str(normalized["game_id"]))
                    fingerprints.add(normalized["game_fingerprint"])
                    records.append(normalized)
                except (ValueError, TypeError, KeyError) as exc:
                    reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                    rejected[reason] += 1
                    rejected_stream.write(json.dumps({"line": line_index, "game_id": record.get("game_id") if isinstance(record, dict) else None,
                                                       "reason": reason}) + "\n")
        manifest.update(accepted_games=len(records), rejected_counts=dict(rejected))
        manifest["split"] = assign_splits(records)
        for split in SPLITS:
            selected = [record for record in records if record["split"] == split]
            manifest["splits"][split] = {"games": len(selected), "positions": 0, "policy_positions": 0,
                                         "results": dict(Counter(row["result"] for row in selected)),
                                         "terminations": dict(Counter(str(row["termination"]) for row in selected)),
                                         "first_end_time": selected[0]["end_time"], "last_end_time": selected[-1]["end_time"]}
        with (output_dir / "games.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        manifest["games_sha256"] = _sha256(output_dir / "games.jsonl")
        for split in SPLITS:
            buffers = {name: [] for name in ("states", "actions", "values", "policy_mask", "game_indices", "plies")}
            buffered, shard_index = 0, 0

            def flush():
                nonlocal buffered, shard_index
                if not buffered:
                    return
                arrays = {key: np.concatenate(parts) for key, parts in buffers.items()}
                name = f"{split}-{shard_index:05d}.npz"
                np.savez_compressed(output_dir / name, **arrays)
                manifest["shards"].append({"path": name, "split": split, "positions": buffered,
                                            "policy_positions": int(arrays["policy_mask"].sum()),
                                            "sha256": _sha256(output_dir / name)})
                for parts in buffers.values():
                    parts.clear()
                buffered = 0
                shard_index += 1

            for record in records:
                if record["split"] != split:
                    continue
                arrays = encode_record(record)
                count = len(arrays["actions"])
                manifest["splits"][split]["positions"] += count
                manifest["splits"][split]["policy_positions"] += int(arrays["policy_mask"].sum())
                for start in range(0, count, shard_size):
                    stop = min(count, start + shard_size)
                    if buffered and buffered + stop - start > shard_size:
                        flush()
                    for name, values in arrays.items():
                        buffers[name].append(values[start:stop])
                    buffered += stop - start
                    if buffered >= shard_size:
                        flush()
                if (record["game_index"] + 1) % 100 == 0:
                    print(f"Prepared {record['game_index'] + 1}/{len(records)} games", flush=True)
            flush()
        manifest["status"] = "completed_with_rejections" if rejected else "completed"
        _write_json(output_dir / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(output_dir / "manifest.json", manifest)
        raise
    return manifest


def load_manifest(dataset_dir):
    manifest = json.loads((Path(dataset_dir) / "manifest.json").read_text())
    if manifest.get("status") not in {"completed", "completed_with_rejections"}:
        raise ValueError("Prepared dataset is incomplete or failed")
    return manifest


def iter_game_records(dataset_dir, split=None):
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    manifest = load_manifest(dataset_dir)
    path = Path(dataset_dir) / "games.jsonl"
    if _sha256(path) != manifest.get("games_sha256"):
        raise ValueError("Prepared game records do not match their recorded SHA256")
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if split is None or record["split"] == split:
                yield record


def iter_batches(dataset_dir, split, batch_size, seed=1, shuffle=False):
    """Yield CPU tensors, loading at most one compact shard plus a minibatch.

    Shard order and rows within each shard are shuffled with a private generator;
    this preserves deterministic runs without touching global RNG state.
    """
    if split not in SPLITS or batch_size < 1:
        raise ValueError("A valid split and positive batch_size are required")
    root = Path(dataset_dir)
    manifest = load_manifest(root)
    shards = [row for row in manifest["shards"] if row["split"] == split]
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(shards)
    for shard in shards:
        path = root / shard["path"]
        if _sha256(path) != shard.get("sha256"):
            raise ValueError(f"Prepared shard does not match its recorded SHA256: {shard['path']}")
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        size = len(arrays["actions"])
        if size != shard["positions"] or arrays["states"].shape != (size, 20, 8, 8):
            raise ValueError(f"Corrupt shard dimensions: {shard['path']}")
        indices = rng.permutation(size) if shuffle else np.arange(size)
        for start in range(0, size, batch_size):
            chosen = indices[start:start + batch_size]
            states = torch.from_numpy(arrays["states"][chosen].astype(np.float32))
            states[:, 18].div_(100)
            states[:, 19].div_(3)
            actions = torch.from_numpy(arrays["actions"][chosen].astype(np.int64))
            if ((actions < 0) | (actions >= NUM_ACTIONS)).any():
                raise ValueError(f"Invalid action in shard: {shard['path']}")
            yield {"states": states, "actions": actions,
                   "values": torch.from_numpy(arrays["values"][chosen].astype(np.float32)),
                   "policy_mask": torch.from_numpy(arrays["policy_mask"][chosen].astype(np.bool_)),
                   "game_indices": torch.from_numpy(arrays["game_indices"][chosen].astype(np.int64)),
                   "plies": torch.from_numpy(arrays["plies"][chosen].astype(np.int64))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=8192)
    args = parser.parse_args(argv)
    manifest = prepare_dataset(args.input, args.output_dir, args.shard_size)
    print(json.dumps({key: manifest[key] for key in ("status", "input_records", "accepted_games", "rejected_counts", "splits")}, indent=2))


if __name__ == "__main__":
    main()
