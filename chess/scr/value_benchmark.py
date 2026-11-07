"""Freeze outcome-labelled self-play games for evaluation, separate from replay.

The fixed game budget retains every generated game. Move-limit games retain
their histories but have unknown results and contribute no value labels. Known
draws are genuine outcomes and remain eligible. Games use independent per-game
seeds; timing metadata is not expected to reproduce byte for byte.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import torch

from .chess_engine import GameState
from .experiment import write_json
from .rl_interface import load_model
from .self_play import play_self_game, terminal_details


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def game_record(info, white_result, *, game_index, seed, benchmark_id, max_plies):
    """Verify the complete history and keep unknown outcomes distinct from zero."""
    state = GameState.from_fen(info['initial_fen'])
    for uci in info['moves']:
        if terminal_details(state) is not None:
            raise ValueError('Benchmark history continues after a terminal position')
        state.push_uci(uci)
    if info['plies'] != len(info['moves']) or state.to_fen() != info['final_fen']:
        raise ValueError('Benchmark history does not match its final FEN or ply count')
    ending = terminal_details(state)
    if info['truncated']:
        if (ending is not None or info['termination'] != 'move_limit' or info['outcome'] != '*'
                or len(info['moves']) != max_plies or white_result != 0):
            raise ValueError('Inconsistent unfinished benchmark game')
        result, observed_value = '*', None
    else:
        if ending is None or ending != (info['termination'], white_result):
            raise ValueError('Completed benchmark outcome must match the terminal board')
        result = '1-0' if white_result > 0 else '0-1' if white_result < 0 else '1/2-1/2'
        if info['outcome'] != result:
            raise ValueError('Benchmark result disagrees with the observed outcome')
        observed_value = white_result
    return {
        'game_id': f'{benchmark_id}:{game_index:04d}', 'game_index': game_index, 'seed': seed,
        'initial_fen': info['initial_fen'], 'moves': list(info['moves']),
        'final_fen': info['final_fen'], 'plies': len(info['moves']),
        'result': result, 'white_result': observed_value,
        'termination': info['termination'], 'truncated': bool(info['truncated']),
        'duration_seconds': info['duration_seconds'],
    }


def summarize(records):
    completed = [row for row in records if not row['truncated'] and row['result'] != '*']
    excluded = [row for row in records if row['truncated'] or row['result'] == '*']
    return {
        'games': len(records), 'completed_games': len(completed),
        'unresolved_games': len(excluded),
        'decisive_games': sum(row['result'] in ('1-0', '0-1') for row in completed),
        'actual_draws': sum(row['result'] == '1/2-1/2' for row in completed),
        'positions': sum(row['plies'] for row in records),
        'eligible_value_positions': sum(row['plies'] for row in completed),
        'excluded_value_positions': sum(row['plies'] for row in excluded),
        'result_counts': dict(Counter(row['result'] for row in records)),
        'termination_counts': dict(Counter(row['termination'] for row in records)),
        'excluded_game_reasons': dict(Counter(row['termination'] for row in excluded)),
    }


def generate_benchmark(checkpoint, output_dir, *, games=20, sims=64, temperature=1.0,
                       max_plies=400, seed=20261107, threads=4, device='cpu',
                       root_noise=True, dir_alpha=.3, dir_epsilon=.25):
    """Generate exactly ``games`` frozen-model games without training or replay.

    The output directory must be empty. A failed/interrupted manifest cannot be
    consumed as a completed benchmark. Python/Torch RNG state and thread count
    are restored when this function returns, including on failure.
    """
    for name, value in (('games', games), ('sims', sims), ('max_plies', max_plies), ('threads', threads)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(seed) is not int or seed < 0 or seed + games >= 2 ** 63:
        raise ValueError('seed must allow nonnegative per-game seeds below 2**63')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if not math.isfinite(dir_alpha) or dir_alpha <= 0:
        raise ValueError('dir_alpha must be finite and positive')
    if not math.isfinite(dir_epsilon) or not 0 <= dir_epsilon <= 1:
        raise ValueError('dir_epsilon must be between zero and one')
    if str(device) != 'cpu':
        raise ValueError('Frozen benchmark generation currently supports CPU only')
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Benchmark directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        'games': games, 'sims': sims, 'temperature': temperature, 'max_plies': max_plies,
        'seed': seed, 'seed_policy': 'game seed = seed + zero-based game_index',
        'threads': threads, 'device': 'cpu', 'root_noise': bool(root_noise),
        'dir_alpha': dir_alpha, 'dir_epsilon': dir_epsilon,
        'search_tree_reuse': True, 'search_batch_size': 16,
        'initial_fen': GameState().to_fen(),
    }
    identity = json.dumps({'checkpoint_sha256': checkpoint_hash, 'settings': settings}, sort_keys=True)
    benchmark_id = 'selfplay-value-' + hashlib.sha256(identity.encode()).hexdigest()[:20]
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {str(path.relative_to(source_root)): sha256_file(path)
                     for path in sorted((source_root / 'chess/scr').glob('*.py'))}
    manifest = {
        'schema_version': 1, 'kind': 'frozen_selfplay_value_benchmark',
        'status': 'running', 'benchmark_id': benchmark_id, 'started_at': utc_now(),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
        'settings': settings, 'source_sha256': source_hashes,
        'torch_version': str(torch.__version__), 'python_version': platform.python_version(),
        'games_file': 'games.jsonl', 'games_sha256': None, 'counts': summarize([]),
        'purpose': 'Held-out observed-outcome value evaluation; never add these games to training replay.',
        'value_perspective': 'player to move at every pre-move position',
        'eligibility': 'Use all positions in completed games with known outcomes; exclude entire truncated/result-* games; retain genuine draws.',
        'limitations': [
            'Labels are outcomes under the frozen baseline self-play policy, not minimax values.',
            'Positions within one game are correlated; report game-level uncertainty.',
            'A fixed game budget may yield few completed games or decisive outcomes; report counts without replacing games selectively.',
        ],
    }
    write_json(output_dir / 'manifest.json', manifest)
    games_path = output_dir / 'games.jsonl'
    games_path.touch(exist_ok=False)
    records, started = [], time.monotonic()
    python_state, torch_state, old_threads = random.getstate(), torch.get_rng_state(), torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        model = load_model(checkpoint, device='cpu')
        model.eval()
        with torch.inference_mode():
            for game_index in range(games):
                game_seed = seed + game_index
                random.seed(game_seed)
                torch.manual_seed(game_seed)
                samples, result, info = play_self_game(
                    GameState(), model, num_mcts_sims=sims, temperature=temperature,
                    max_moves=max_plies, add_root_dirichlet=root_noise,
                    dir_alpha=dir_alpha, dir_epsilon=dir_epsilon,
                    return_info=True, record_trace=False,
                )
                # Dense policy/state tensors are transient for this one game.
                del samples
                record = game_record(info, result, game_index=game_index, seed=game_seed,
                                     benchmark_id=benchmark_id, max_plies=max_plies)
                with games_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                records.append(record)
                manifest.update(counts=summarize(records), updated_at=utc_now(),
                                duration_seconds=time.monotonic() - started,
                                games_sha256=sha256_file(games_path))
                write_json(output_dir / 'manifest.json', manifest)
                print(f"Benchmark {game_index + 1}/{games}: {record['termination']}, "
                      f"{record['plies']} plies; completed {manifest['counts']['completed_games']}, "
                      f"unresolved {manifest['counts']['unresolved_games']}", flush=True)
        if sha256_file(checkpoint) != checkpoint_hash:
            raise ValueError('Source checkpoint changed during benchmark generation')
        manifest.update(status='completed', completed_at=utc_now(),
                        duration_seconds=time.monotonic() - started)
        write_json(output_dir / 'manifest.json', manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}', updated_at=utc_now(),
                        counts=summarize(records), games_sha256=sha256_file(games_path),
                        duration_seconds=time.monotonic() - started)
        write_json(output_dir / 'manifest.json', manifest)
        raise
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--sims', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--max-plies', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--no-root-noise', action='store_true')
    parser.add_argument('--dir-alpha', type=float, default=.3)
    parser.add_argument('--dir-epsilon', type=float, default=.25)
    args = vars(parser.parse_args(argv))
    args['root_noise'] = not args.pop('no_root_noise')
    manifest = generate_benchmark(**args)
    print(json.dumps({'status': manifest['status'], 'counts': manifest['counts'],
                      'games_sha256': manifest['games_sha256']}, indent=2))


if __name__ == '__main__':
    main()
