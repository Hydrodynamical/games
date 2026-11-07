"""Download a reproducible selection of public Chess.com games, using serial requests.

Usage: python chess/scr/download_games.py --username Hikaru --limit 5000
       --output-dir data/hikaru-blitz-5000

Only documented public API endpoints are used. Cached JSON responses are retained
with source URLs and hashes; PGN legality is checked by the dataset preparation
stage, not by this dependency-free downloader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


API = "https://api.chess.com/pub/player"
USER_AGENT = "HydrodynamicalChessResearch/1.0 (+https://github.com/Hydrodynamical/games)"
RESULT_RE = re.compile(r'^\[Result\s+"(1-0|0-1|1/2-1/2|\*)"\]\s*$', re.MULTILINE)
HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\]\s*$', re.MULTILINE)
LOSS_CODES = {"checkmated", "resigned", "timeout", "lose", "abandoned"}
DRAW_CODES = {"agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def _retry_delay(header, attempt):
    """Honor Retry-After up to 60 seconds; otherwise use bounded backoff."""
    delay = min(2 ** attempt, 30)
    if header:
        try:
            requested = float(header)
        except ValueError:
            try:
                requested = (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                requested = delay
        if math.isfinite(requested):
            delay = min(60, max(0, requested))
    return delay


class CachedJSONClient:
    """A sequential, hash-checked JSON cache with optional HTTP revalidation."""

    def __init__(self, directory, *, refresh=False, offline=False, timeout=30, retries=3,
                 opener=None, sleep=time.sleep):
        if refresh and offline:
            raise ValueError("--refresh and --offline cannot be combined")
        if not math.isfinite(timeout) or timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries must be nonnegative")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.refresh, self.offline = refresh, offline
        self.timeout, self.retries = timeout, retries
        self.opener, self.sleep = opener or urlopen, sleep
        self.sources = []

    def get(self, url, name):
        path = self.directory / (name + ".json")
        metadata_path = self.directory / (name + ".meta.json")
        cached, metadata = None, None
        if path.exists() or metadata_path.exists():
            if not path.is_file() or not metadata_path.is_file():
                raise ValueError(f"Incomplete cache entry: {path}; remove it or use a new output directory")
            cached = path.read_bytes()
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("url") != url or metadata.get("sha256") != hashlib.sha256(cached).hexdigest():
                raise ValueError(f"Cache integrity check failed: {path}")
            if not self.refresh:
                self.sources.append(dict(metadata, cache_file=str(path.name), from_cache=True))
                return json.loads(cached)
        if self.offline:
            raise FileNotFoundError(f"No cached response for {url}; run once without --offline")
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if metadata:
            if metadata.get("etag"):
                headers["If-None-Match"] = metadata["etag"]
            if metadata.get("last_modified"):
                headers["If-Modified-Since"] = metadata["last_modified"]
        request = Request(url, headers=headers)
        for attempt in range(self.retries + 1):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    content = response.read()
                    value = json.loads(content)
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected a JSON object from {url}")
                    metadata = {"url": url, "fetched_at": utc_now(),
                                "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                                "etag": response.headers.get("ETag"),
                                "last_modified": response.headers.get("Last-Modified")}
                    _atomic_bytes(path, content)
                    _write_json(metadata_path, metadata)
                    self.sources.append(dict(metadata, cache_file=str(path.name), from_cache=False))
                    return value
            except HTTPError as exc:
                if exc.code == 304 and cached is not None:
                    self.sources.append(dict(metadata, cache_file=str(path.name), from_cache=True,
                                             revalidated_at=utc_now()))
                    return json.loads(cached)
                retryable = exc.code == 429 or exc.code in {500, 502, 503, 504}
                if not retryable or attempt >= self.retries:
                    raise
                self.sleep(_retry_delay(exc.headers.get("Retry-After") if exc.headers else None, attempt))
            except (URLError, TimeoutError, ConnectionError):
                if attempt >= self.retries:
                    raise
                self.sleep(_retry_delay(None, attempt))


def _normalise_game(game, username, archive_url):
    """Return (portable record, None), or (None, stable rejection reason)."""
    if not isinstance(game, dict):
        return None, "malformed_game"
    if game.get("rules") != "chess":
        return None, "nonstandard_rules"
    if game.get("time_class") != "blitz":
        return None, "not_blitz"
    if game.get("rated") is not True:
        return None, "not_rated"
    white, black = game.get("white", {}), game.get("black", {})
    if not isinstance(white, dict) or not isinstance(black, dict):
        return None, "missing_players"
    white_name, black_name = white.get("username", ""), black.get("username", "")
    if not isinstance(white_name, str) or not isinstance(black_name, str) or not white_name or not black_name:
        return None, "missing_players"
    matches = [color for color, name in (("w", white_name), ("b", black_name)) if name.casefold() == username]
    if len(matches) != 1:
        return None, "wrong_player"
    pgn = game.get("pgn")
    if not isinstance(pgn, str) or not pgn.strip():
        return None, "missing_pgn"
    result_match = RESULT_RE.findall(pgn)
    if len(result_match) != 1 or result_match[0] == "*":
        return None, "unknown_result"
    result = result_match[0]
    headers = dict(HEADER_RE.findall(pgn))
    if (headers.get("White", "").casefold() != white_name.casefold()
            or headers.get("Black", "").casefold() != black_name.casefold()):
        return None, "player_header_mismatch"
    white_result, black_result = white.get("result"), black.get("result")
    expected = ("1-0" if white_result == "win" and black_result in LOSS_CODES
                else "0-1" if black_result == "win" and white_result in LOSS_CODES
                else "1/2-1/2" if white_result in DRAW_CODES and black_result in DRAW_CODES else None)
    if result != expected:
        return None, "result_mismatch"
    end_time = game.get("end_time")
    if not isinstance(end_time, int) or isinstance(end_time, bool) or end_time <= 0:
        return None, "missing_end_time"
    url = game.get("url", "")
    if not isinstance(url, str):
        return None, "missing_game_url"
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.chess.com", "chess.com"} or not parsed.path.startswith("/game/"):
        return None, "missing_game_url"
    # The final path component is the public game ID. Prefix it with game type
    # because legacy live/daily numeric namespaces need not be interchangeable.
    game_id = parsed.path.strip("/")
    return {
        "game_id": game_id, "url": url, "end_time": end_time,
        "white": white, "black": black, "target_color": matches[0], "result": result,
        "termination": headers.get("Termination", ""),
        "white_result": white_result, "black_result": black_result,
        "time_control": game.get("time_control"), "time_class": "blitz", "rated": True,
        "rules": "chess", "pgn": pgn, "archive_url": archive_url,
    }, None


def download_dataset(username, limit, output_dir, *, refresh=False, offline=False, timeout=30,
                     retries=3, opener=None, sleep=time.sleep):
    """Select the newest eligible games, write them chronologically, return manifest.

    Requests are strictly serial. Each monthly JSON archive contains PGNs plus
    authoritative speed/rules/rated metadata. If too few eligible games exist,
    write a partial selection and failed manifest, then raise ValueError.
    """
    if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("username must contain only letters, digits, hyphens, or underscores")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    username = username.casefold()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    existing_manifest = directory / "manifest.json"
    preserve_completed = False
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())
        if previous.get("username") != username:
            raise ValueError("Output directory belongs to a different player; choose a new directory")
        preserve_completed = previous.get("status") == "completed" and (directory / "games.jsonl").is_file()
    client = CachedJSONClient(directory / "cache", refresh=refresh, offline=offline,
                              timeout=timeout, retries=retries, opener=opener, sleep=sleep)
    manifest = {"schema_version": 1, "status": "downloading", "started_at": utc_now(),
                "username": username, "requested_games": limit, "selected_games": 0,
                "source": "Chess.com Published-Data API", "user_agent": USER_AGENT,
                "selection": {"order": "newest eligible games; output oldest first", "time_class": "blitz",
                              "rules": "chess", "rated": True, "known_result": True,
                              "colors": ["w", "b"], "outcomes": ["1-0", "0-1", "1/2-1/2"]},
                "profile": None, "archives_visited": [], "rejection_counts": {}, "sources": []}
    selected, seen, rejected = [], set(), Counter()

    def finish(status, error=None):
        chosen = sorted(selected, key=lambda row: (row["end_time"], row["game_id"]))[-limit:]
        content = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in chosen).encode()
        keep_previous = preserve_completed and status != "completed"
        _atomic_bytes(directory / ("games.partial.jsonl" if keep_previous else "games.jsonl"), content)
        manifest.update(status=status, updated_at=utc_now(), selected_games=len(chosen),
                        rejection_counts=dict(rejected), sources=client.sources, error=error,
                        games_sha256=hashlib.sha256(content).hexdigest(), games_bytes=len(content),
                        outcome_counts=dict(Counter(row["result"] for row in chosen)),
                        target_color_counts=dict(Counter(row["target_color"] for row in chosen)))
        manifest["date_span"] = ({"first_end_time": chosen[0]["end_time"], "last_end_time": chosen[-1]["end_time"],
                                  "first_utc": datetime.fromtimestamp(chosen[0]["end_time"], timezone.utc).isoformat(),
                                  "last_utc": datetime.fromtimestamp(chosen[-1]["end_time"], timezone.utc).isoformat()}
                                 if chosen else None)
        _write_json(directory / "manifest.failed.json" if keep_previous else existing_manifest, manifest)
        (directory / "manifest.in-progress.json").unlink(missing_ok=True)

    _write_json(directory / "manifest.in-progress.json" if preserve_completed else existing_manifest, manifest)
    try:
        profile = client.get(f"{API}/{username}", "profile")
        if str(profile.get("username", "")).casefold() != username:
            raise ValueError("Profile username does not match the requested player")
        manifest["profile"] = {key: profile.get(key) for key in ("username", "name", "title", "player_id", "url")}
        listing = client.get(f"{API}/{username}/games/archives", "archives")
        archives = listing.get("archives")
        if not isinstance(archives, list) or not all(isinstance(url, str) for url in archives):
            raise ValueError("Archive response must contain a list of URLs")
        pattern = re.compile(r"/pub/player/" + re.escape(username) + r"/games/(\d{4})/(0[1-9]|1[0-2])$")
        validated = []
        for url in set(archives):
            parsed = urlsplit(url)
            match = pattern.fullmatch(parsed.path)
            if parsed.scheme != "https" or parsed.netloc != "api.chess.com" or parsed.query or parsed.fragment or not match:
                raise ValueError(f"Unexpected archive URL: {url}")
            validated.append((match.group(1) + "-" + match.group(2), url))
        for month, url in sorted(validated, reverse=True):
            payload = client.get(url, month)
            games = payload.get("games")
            if not isinstance(games, list):
                raise ValueError(f"Archive has no game list: {url}")
            manifest["archives_visited"].append(url)
            for game in games:
                row, reason = _normalise_game(game, username, url)
                if reason:
                    rejected[reason] += 1
                    continue
                if row["game_id"] in seen:
                    rejected["duplicate_game"] += 1
                    continue
                seen.add(row["game_id"])
                selected.append(row)
            print(f"{month}: {len(games)} records; {len(selected)} eligible unique blitz games", flush=True)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            raise ValueError(f"Requested {limit} eligible games, but only {len(selected)} were available")
        finish("completed")
        return manifest
    except BaseException as exc:
        finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", f"{type(exc).__name__}: {exc}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="Revalidate cached responses with the API")
    parser.add_argument("--offline", action="store_true", help="Only use cached responses")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        manifest = download_dataset(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(json.dumps({"status": manifest["status"], "selected_games": manifest["selected_games"],
                      "date_span": manifest["date_span"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
