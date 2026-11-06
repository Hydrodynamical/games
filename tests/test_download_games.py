"""Public-game downloads filter, cache, and fail explicitly without live HTTP."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
from scr.download_games import API, CachedJSONClient, _normalise_game, _retry_delay, download_dataset


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {"ETag": '"example"', "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT"}

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def game(number=1, *, end_time=None, color="w", result="1-0", **changes):
    white, black = ("Hikaru", "Opponent") if color == "w" else ("Opponent", "hIKARU")
    white_result, black_result = {"1-0": ("win", "resigned"), "0-1": ("timeout", "win"),
                                  "1/2-1/2": ("agreed", "agreed"), "*": (None, None)}[result]
    data = {"url": f"https://www.chess.com/game/live/{number}", "end_time": end_time or 1700000000 + number,
            "white": {"username": white, "result": white_result, "rating": 3000},
            "black": {"username": black, "result": black_result, "rating": 2800},
            "rules": "chess", "time_class": "blitz", "rated": True, "time_control": "180",
            "pgn": f'[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n'
                   f'[Termination "Example resignation"]\n\n1. e4 e5 {result}\n'}
    data.update(changes)
    return data


class DownloadTests(unittest.TestCase):
    def fixture(self, recent, older=None):
        newest = f"{API}/hikaru/games/2025/02"
        old = f"{API}/hikaru/games/2025/01"
        payloads = {f"{API}/hikaru": {"username": "Hikaru", "name": "Hikaru Nakamura", "title": "GM"},
                    f"{API}/hikaru/games/archives": {"archives": [old, newest]},
                    newest: {"games": recent}, old: {"games": older or []}}
        calls = []

        def opener(request, timeout):
            self.assertGreater(timeout, 0)
            self.assertIn("Hydrodynamical", request.get_header("User-agent"))
            calls.append(request.full_url)
            return Response(payloads[request.full_url])

        return opener, payloads, calls

    def test_selects_latest_unique_eligible_games_and_writes_chronologically(self):
        opener, _, calls = self.fixture([game(4), game(2, color="b", result="0-1"),
                                         game(4), game(3, result="1/2-1/2"), game(5, rated=False),
                                         game(6, time_class="bullet"), game(7, rules="chess960")])
        with tempfile.TemporaryDirectory() as temporary:
            manifest = download_dataset("HiKaRu", 3, temporary, opener=opener)
            rows = [json.loads(line) for line in (Path(temporary) / "games.jsonl").read_text().splitlines()]
            self.assertEqual([row["game_id"] for row in rows], ["game/live/2", "game/live/3", "game/live/4"])
            self.assertEqual([row["target_color"] for row in rows], ["b", "w", "w"])
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["selected_games"], 3)
            self.assertEqual(manifest["outcome_counts"], {"0-1": 1, "1/2-1/2": 1, "1-0": 1})
            self.assertEqual(manifest["rejection_counts"], {"duplicate_game": 1, "not_rated": 1,
                                                          "not_blitz": 1, "nonstandard_rules": 1})
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(source["sha256"] and source["url"] for source in manifest["sources"]))

    def test_old_months_backfill_and_surplus_is_trimmed_by_end_time(self):
        opener, _, calls = self.fixture([game(5)], [game(1), game(4), game(2), game(3)])
        with tempfile.TemporaryDirectory() as temporary:
            manifest = download_dataset("hikaru", 3, temporary, opener=opener)
            rows = [json.loads(line) for line in (Path(temporary) / "games.jsonl").read_text().splitlines()]
            self.assertEqual([row["game_id"] for row in rows], ["game/live/3", "game/live/4", "game/live/5"])
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(manifest["archives_visited"]), 2)

    def test_cached_and_offline_runs_do_not_request_http(self):
        opener, _, _ = self.fixture([game(1)])
        with tempfile.TemporaryDirectory() as temporary:
            original = download_dataset("hikaru", 1, temporary, opener=opener)

            def forbidden(*args, **kwargs):
                self.fail("Cache reuse must not request HTTP")

            repeated = download_dataset("hikaru", 1, temporary, offline=True, opener=forbidden)
            self.assertEqual(original["games_sha256"], repeated["games_sha256"])
            self.assertTrue(all(source["from_cache"] for source in repeated["sources"]))

    def test_not_enough_games_retains_partial_data_and_failed_manifest(self):
        opener, _, _ = self.fixture([game(1)], [game(2, result="*")])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "only 1"):
                download_dataset("hikaru", 3, temporary, opener=opener)
            manifest = json.loads((Path(temporary) / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["selected_games"], 1)
            self.assertEqual(manifest["rejection_counts"], {"unknown_result": 1})
            self.assertEqual(len((Path(temporary) / "games.jsonl").read_text().splitlines()), 1)

    def test_failed_refresh_preserves_completed_selection(self):
        opener, _, _ = self.fixture([game(1)])
        with tempfile.TemporaryDirectory() as temporary:
            download_dataset("hikaru", 1, temporary, opener=opener)
            directory = Path(temporary)
            saved_games = (directory / "games.jsonl").read_bytes()
            saved_manifest = (directory / "manifest.json").read_bytes()

            def unavailable(request, timeout):
                raise HTTPError(request.full_url, 403, "forbidden", {}, io.BytesIO())

            with self.assertRaises(HTTPError):
                download_dataset("hikaru", 1, temporary, refresh=True, opener=unavailable)
            self.assertEqual(saved_games, (directory / "games.jsonl").read_bytes())
            self.assertEqual(saved_manifest, (directory / "manifest.json").read_bytes())
            failure = json.loads((directory / "manifest.failed.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertTrue((directory / "games.partial.jsonl").is_file())
            self.assertFalse((directory / "manifest.in-progress.json").exists())

    def test_result_and_identity_consistency(self):
        row, reason = _normalise_game(game(result="0-1"), "hikaru", "archive")
        self.assertIsNone(reason)
        self.assertEqual(row["white_result"], "timeout")
        bad = game()
        bad["black"]["result"] = "win"
        self.assertEqual(_normalise_game(bad, "hikaru", "archive")[1], "result_mismatch")
        bad = game(pgn='[White "SomeoneElse"]\n[Black "Opponent"]\n[Result "1-0"]\n\n1. e4 1-0')
        self.assertEqual(_normalise_game(bad, "hikaru", "archive")[1], "player_header_mismatch")

    def test_untrusted_archive_url_is_rejected_before_request(self):
        opener, payloads, calls = self.fixture([])
        payloads[f"{API}/hikaru/games/archives"]["archives"] = ["https://evil.example/pub/player/hikaru/games/2025/02"]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unexpected archive URL"):
                download_dataset("hikaru", 1, temporary, opener=opener)
            self.assertEqual(len(calls), 2)

    def test_retry_after_and_http_304_cache_revalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            waits, calls = [], []

            def rate_limited(request, timeout):
                calls.append(request)
                if len(calls) == 1:
                    raise HTTPError(request.full_url, 429, "retry", {"Retry-After": "2"}, io.BytesIO())
                return Response({"ok": True})

            client = CachedJSONClient(temporary, opener=rate_limited, sleep=waits.append)
            self.assertEqual(client.get("https://api.chess.com/example", "example"), {"ok": True})
            self.assertEqual(waits, [2])

            def unchanged(request, timeout):
                self.assertEqual(request.get_header("If-none-match"), '"example"')
                raise HTTPError(request.full_url, 304, "unchanged", {}, io.BytesIO())

            refreshed = CachedJSONClient(temporary, opener=unchanged, refresh=True)
            self.assertEqual(refreshed.get("https://api.chess.com/example", "example"), {"ok": True})
            self.assertTrue(refreshed.sources[0]["from_cache"])
            self.assertEqual(_retry_delay("9999", 0), 60)

    def test_nonretryable_error_and_corrupt_cache_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []

            def forbidden(request, timeout):
                calls.append(request)
                raise HTTPError(request.full_url, 403, "forbidden", {}, io.BytesIO())

            client = CachedJSONClient(temporary, opener=forbidden, sleep=lambda _: self.fail("Must not retry 403"))
            with self.assertRaises(HTTPError):
                client.get("https://api.chess.com/example", "example")
            self.assertEqual(len(calls), 1)
            good = CachedJSONClient(temporary, opener=lambda *a, **k: Response({"ok": True}))
            good.get("https://api.chess.com/example", "example")
            (Path(temporary) / "example.json").write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError, "integrity"):
                good.get("https://api.chess.com/example", "example")


if __name__ == "__main__":
    unittest.main()
