"""Human data preserves history, outcome semantics, and chronological isolation."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chess"))
try:
    import chess as reference
    HAS_CHESS = hasattr(reference, "Board")
except ImportError:
    HAS_CHESS = False
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_CHESS and HAS_TORCH:
    import numpy as np
    import torch
    from scr.chess_engine import GameState
    from scr import human_data
    from scr.rl_interface import encode_gamestate, move_to_action


@unittest.skipUnless(HAS_CHESS and HAS_TORCH, "install requirements/human.txt for supervised data tests")
class HumanDataTests(unittest.TestCase):
    def record(self, index=0, target_color="w", result="1-0", moves="1. e4 e5 2. Nf3 Nc6"):
        white, black = ("Hikaru", "Opponent") if target_color == "w" else ("Opponent", "Hikaru")
        day = f"2026.01.{index + 1:02d}"
        return {"game_id": f"game-{index}", "url": f"https://www.chess.com/game/live/{index}",
                "end_time": int(datetime(2026, 1, index + 1, tzinfo=timezone.utc).timestamp()),
                "white": {"username": white}, "black": {"username": black},
                "target_color": target_color, "result": result,
                "termination": "resigned" if result != "1/2-1/2" else "agreed",
                "rules": "chess", "rated": True, "time_class": "blitz", "time_control": "180",
                "pgn": f'[Event "Live Chess"]\n[Date "{day}"]\n[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n\n{moves} {result}\n'}

    def prepare(self, directory, records=None, shard_size=5):
        records = records if records is not None else [self.record(i, "b" if i % 2 else "w") for i in range(10)]
        source = Path(directory) / "source.jsonl"
        source.write_text("".join(json.dumps(record) + "\n" for record in records))
        output = Path(directory) / "prepared"
        return output, human_data.prepare_dataset(source, output, shard_size)

    def test_known_nonterminal_outcomes_and_player_only_policy(self):
        for color in ("w", "b"):
            record = human_data.parse_record(self.record(target_color=color))
            record["game_index"] = 9
            arrays = human_data.encode_record(record)
            self.assertEqual(arrays["values"].tolist(), [1, -1, 1, -1])
            self.assertEqual(arrays["policy_mask"].tolist(), [color == "w", color == "b"] * 2)
            self.assertEqual(arrays["game_indices"].tolist(), [9] * 4)
        draw = human_data.parse_record(self.record(result="1/2-1/2"))
        draw["game_index"] = 0
        self.assertEqual(human_data.encode_record(draw)["values"].tolist(), [0] * 4)

    def test_history_planes_and_underpromotion_actions_survive_encoding(self):
        repetition = self.record(moves="1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6")
        record = human_data.parse_record(repetition)
        record["game_index"] = 0
        arrays = human_data.encode_record(record)
        self.assertEqual(int(arrays["states"][0, 19, 0, 0]), 1)
        self.assertEqual(int(arrays["states"][4, 19, 0, 0]), 2)
        self.assertEqual(int(arrays["states"][4, 18, 0, 0]), 4)
        promotion = self.record(moves="1. a8=N")
        promotion["pgn"] = promotion["pgn"].replace('[Result "1-0"]', '[Result "1-0"]\n[SetUp "1"]\n[FEN "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]')
        # A knight-only ending is a rules-defined draw, so its recorded result
        # must be draw even if normal nonterminal resignations are accepted.
        promotion["result"] = "1/2-1/2"
        promotion["pgn"] = promotion["pgn"].replace('1-0', '1/2-1/2')
        normalized = human_data.parse_record(promotion)
        normalized["game_index"] = 0
        arrays = human_data.encode_record(normalized)
        gs = GameState.from_fen(normalized["initial_fen"])
        self.assertEqual(arrays["actions"].tolist(), [move_to_action(gs.parse_uci("a7a8n"))])
        self.assertGreaterEqual(int(arrays["actions"][0]), 4096)

    def test_bad_or_partial_pgn_and_metadata_are_rejected(self):
        examples = []
        unknown = self.record(); unknown["result"] = "*"; examples.append(unknown)
        incomplete = self.record(); incomplete["pgn"] = incomplete["pgn"].rsplit("1-0", 1)[0]; examples.append(incomplete)
        mismatch = self.record(); mismatch["white"] = "SomeoneElse"; examples.append(mismatch)
        multiple = self.record(); multiple["pgn"] += "\n" + self.record(1)["pgn"]; examples.append(multiple)
        variant = self.record(); variant["rules"] = "chess960"; examples.append(variant)
        illegal = self.record(moves="1. e4 e5 2. e4"); examples.append(illegal)
        terminal_mismatch = self.record(moves="1. f3 e5 2. g4 Qh4#"); examples.append(terminal_mismatch)
        for record in examples:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    human_data.parse_record(record)

    def test_chronological_whole_date_splits_and_deduplication(self):
        records = [self.record(i) for i in reversed(range(10))]
        duplicate = deepcopy(records[0]); duplicate["game_id"] = "different-url-same-game"
        records.extend([deepcopy(records[1]), duplicate])
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = self.prepare(directory, records)
            self.assertEqual(manifest["status"], "completed_with_rejections")
            self.assertEqual(manifest["input_records"], 12)
            self.assertEqual(manifest["accepted_games"], 10)
            self.assertEqual(manifest["rejected_counts"], {"duplicate_game_id": 1, "duplicate_game_fingerprint": 1})
            self.assertEqual([manifest["splits"][s]["games"] for s in human_data.SPLITS], [8, 1, 1])
            split_records = {s: list(human_data.iter_game_records(output, s)) for s in human_data.SPLITS}
            self.assertLess(split_records["train"][-1]["end_time"], split_records["validation"][0]["end_time"])
            self.assertLess(split_records["validation"][-1]["end_time"], split_records["test"][0]["end_time"])
            self.assertEqual(set(r["game_id"] for rows in split_records.values() for r in rows), {f"game-{i}" for i in range(10)})

    def test_date_boundaries_keep_sessions_together(self):
        records = [human_data.parse_record(self.record(i)) for i in range(10)]
        for index in (7, 8):
            records[index]["end_time"] = records[6]["end_time"] + index
        human_data.assign_splits(records)
        self.assertEqual(len({r["split"] for r in records if r["end_time"] // 86400 == records[6]["end_time"] // 86400}), 1)
        with self.assertRaisesRegex(ValueError, "three UTC dates"):
            human_data.assign_splits(records[:2])

    def test_minibatches_roundtrip_and_shuffling_are_bounded_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = self.prepare(directory)
            self.assertTrue(all(s["positions"] <= 5 for s in manifest["shards"]))
            batches = list(human_data.iter_batches(output, "train", 3))
            self.assertEqual(sum(len(batch["actions"]) for batch in batches), 32)
            first = batches[0]
            self.assertTrue(torch.equal(first["states"][0], encode_gamestate(GameState())))
            self.assertEqual(first["states"].dtype, torch.float32)
            self.assertEqual(first["actions"].dtype, torch.int64)
            self.assertEqual(first["policy_mask"].dtype, torch.bool)
            self.assertEqual(first["plies"].tolist(), [0, 1, 2])
            def order(seed):
                return [(int(g), int(p)) for batch in human_data.iter_batches(output, "train", 3, seed, True)
                        for g, p in zip(batch["game_indices"], batch["plies"])]
            before = np.random.get_state()
            self.assertEqual(order(10), order(10))
            self.assertNotEqual(order(10), order(11))
            self.assertEqual(sorted(order(10)), sorted(order(11)))
            after = np.random.get_state()
            self.assertTrue(np.array_equal(before[1], after[1]))
            with self.assertRaises(FileExistsError):
                human_data.prepare_dataset(Path(directory) / "source.jsonl", output)

    def test_failed_preparation_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.prepare(directory, [self.record()])
            output = Path(directory) / "prepared"
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")
            with self.assertRaisesRegex(ValueError, "incomplete or failed"):
                human_data.load_manifest(output)

    def test_modified_games_and_shards_fail_integrity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            output, manifest = self.prepare(directory)
            path = output / manifest["shards"][0]["path"]
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                list(human_data.iter_batches(output, "train", 8))
            path = output / "games.jsonl"
            with path.open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                list(human_data.iter_game_records(output, "test"))


if __name__ == "__main__":
    unittest.main()
