"""A frozen outcome benchmark must not turn unfinished games into draw labels."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'chess'))
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from scr.chess_engine import GameState
    from scr.rl_interface import NUM_ACTIONS
    from scr import value_benchmark as benchmark

    class UniformModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, states):
            return torch.zeros(len(states), NUM_ACTIONS), torch.zeros(len(states), 1)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is an optional training dependency')
class ValueBenchmarkTests(unittest.TestCase):
    def checkpoint(self, directory):
        # The network loader is substituted with a small deterministic model;
        # checkpoint bytes still participate in all provenance/hash checks.
        path = Path(directory) / 'checkpoint.pt'
        path.write_bytes(b'frozen-test-checkpoint')
        return path

    def history(self, moves, termination, outcome, value):
        state = GameState()
        initial = state.to_fen()
        for uci in moves:
            state.push_uci(uci)
        return {'initial_fen': initial, 'moves': moves, 'final_fen': state.to_fen(),
                'plies': len(moves), 'termination': termination, 'outcome': outcome,
                'truncated': termination == 'move_limit', 'result': value, 'duration_seconds': .1}

    def normalize(self, info, value, index=0):
        return benchmark.game_record(info, value, game_index=index, seed=7 + index,
                                     benchmark_id='test', max_plies=2)

    def test_real_search_is_repeatable_and_unknown_results_stay_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory)
            random.seed(123)
            torch.manual_seed(456)
            rng, torch_rng, threads = random.getstate(), torch.get_rng_state().clone(), torch.get_num_threads()
            outputs = []
            with patch.object(benchmark, 'load_model', side_effect=lambda *a, **k: UniformModel()):
                for name in ('one', 'two'):
                    output = Path(directory) / name
                    manifest = benchmark.generate_benchmark(checkpoint, output, games=2, sims=2,
                                                            max_plies=3, seed=17, threads=1)
                    rows = [json.loads(line) for line in (output / 'games.jsonl').read_text().splitlines()]
                    outputs.append(rows)
                    self.assertEqual(manifest['status'], 'completed')
                    self.assertEqual(manifest['counts']['games'], 2)
                    self.assertEqual(manifest['counts']['completed_games'], 0)
                    self.assertEqual(manifest['counts']['actual_draws'], 0)
                    self.assertEqual(manifest['counts']['eligible_value_positions'], 0)
                    self.assertEqual(manifest['counts']['excluded_value_positions'], 6)
                    self.assertEqual(manifest['games_sha256'], benchmark.sha256_file(output / 'games.jsonl'))
                    self.assertEqual(manifest['checkpoint_sha256'], benchmark.sha256_file(checkpoint))
                    self.assertEqual(manifest['settings']['seed'], 17)
                    self.assertTrue(manifest['source_sha256'])
                    for index, row in enumerate(rows):
                        self.assertEqual(row['seed'], 17 + index)
                        self.assertEqual(row['result'], '*')
                        self.assertIsNone(row['white_result'])
                        self.assertTrue(row['truncated'])
                        self.assertNotIn('states', row)
                        self.assertNotIn('policy', row)
            self.assertEqual(random.getstate(), rng)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(torch.get_num_threads(), threads)
            for first, second in zip(*outputs):
                first.pop('duration_seconds'); second.pop('duration_seconds')
                self.assertEqual(first, second)

    def test_decisive_and_genuine_draw_games_remain_eligible(self):
        mate = self.history(['f2f3', 'e7e5', 'g2g4', 'd8h4'], 'checkmate', '0-1', -1)
        repetition = self.history(['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2,
                                  'threefold_repetition', '1/2-1/2', 0)
        cutoff = self.history(['e2e4', 'e7e5'], 'move_limit', '*', 0)
        records = [self.normalize(mate, -1), self.normalize(repetition, 0, 1), self.normalize(cutoff, 0, 2)]
        counts = benchmark.summarize(records)
        self.assertEqual(counts['completed_games'], 2)
        self.assertEqual(counts['actual_draws'], 1)
        self.assertEqual(counts['decisive_games'], 1)
        self.assertEqual(counts['eligible_value_positions'], 12)
        self.assertEqual(counts['excluded_value_positions'], 2)
        self.assertEqual(records[0]['white_result'], -1)
        self.assertEqual(records[1]['white_result'], 0)
        self.assertIsNone(records[2]['white_result'])

    def test_corrupt_or_nonterminal_completed_histories_are_rejected(self):
        cutoff = self.history(['e2e4', 'e7e5'], 'move_limit', '*', 0)
        examples = []
        incorrect_fen = deepcopy(cutoff); incorrect_fen['final_fen'] = incorrect_fen['initial_fen']; examples.append(incorrect_fen)
        incorrect_ply = deepcopy(cutoff); incorrect_ply['plies'] = 1; examples.append(incorrect_ply)
        invented_draw = deepcopy(cutoff); invented_draw.update(truncated=False, outcome='1/2-1/2', termination='stalemate'); examples.append(invented_draw)
        mislabeled_cutoff = deepcopy(cutoff); mislabeled_cutoff['outcome'] = '1/2-1/2'; examples.append(mislabeled_cutoff)
        for info in examples:
            with self.subTest(info=info):
                with self.assertRaises(ValueError):
                    self.normalize(info, 0)
        mate = self.history(['f2f3', 'e7e5', 'g2g4', 'd8h4'], 'checkmate', '0-1', -1)
        with self.assertRaises(ValueError):
            self.normalize(mate, 1)
        mate['moves'].append('a2a3')
        with self.assertRaisesRegex(ValueError, 'continues after'):
            self.normalize(mate, -1)

    def test_failure_records_partial_status_and_restores_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, output = self.checkpoint(directory), Path(directory) / 'failed'
            rng, torch_rng, threads = random.getstate(), torch.get_rng_state().clone(), torch.get_num_threads()
            with patch.object(benchmark, 'load_model', return_value=UniformModel()):
                with patch.object(benchmark, 'play_self_game', side_effect=RuntimeError('search failed')):
                    with self.assertRaisesRegex(RuntimeError, 'search failed'):
                        benchmark.generate_benchmark(checkpoint, output, games=1, threads=1)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['counts']['games'], 0)
            self.assertEqual(manifest['games_sha256'], benchmark.sha256_file(output / 'games.jsonl'))
            self.assertEqual(random.getstate(), rng)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            self.assertEqual(torch.get_num_threads(), threads)
            with self.assertRaises(FileExistsError):
                benchmark.generate_benchmark(checkpoint, output, games=1)

    def test_invalid_settings_cannot_create_a_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, output = self.checkpoint(directory), Path(directory) / 'invalid'
            for kwargs in ({'games': 0}, {'sims': 0}, {'max_plies': 0}, {'threads': 0},
                           {'temperature': float('nan')}, {'seed': -1}, {'device': 'mps'},
                           {'dir_alpha': 0}, {'dir_epsilon': 2}):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        benchmark.generate_benchmark(checkpoint, output, **kwargs)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
