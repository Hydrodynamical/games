"""Train the existing policy/value model on recorded human games.

Policy imitation uses only the selected player's decisions; value supervision
uses observed outcomes from both sides. Dense states exist only per minibatch.
Run with ``PYTHONPATH=chess python -m scr.supervised --help``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch
import torch.nn.functional as F

from .human_data import iter_batches, load_manifest
from .rl_interface import PolicyValueNet, save_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def supervised_step(model, optimizer, batch, device='cpu', value_weight=1.0):
    """One update: player-only move cross entropy plus both-sides value MSE."""
    if not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError('value_weight must be finite and nonnegative')
    model.train()
    states = batch['states'].to(device)
    actions = batch['actions'].to(device)
    targets = batch['values'].to(device)
    mask = batch['policy_mask'].to(device=device, dtype=torch.bool)
    if len(states) == 0 or actions.shape != targets.shape or actions.shape != mask.shape or actions.ndim != 1 or len(actions) != len(states):
        raise ValueError('Batch vectors must have one entry per state')
    if not torch.isfinite(states).all() or not torch.isfinite(targets).all() or (targets.abs() > 1).any():
        raise ValueError('States/targets must be finite and outcome targets in [-1, 1]')
    logits, values = model(states)
    values = values.flatten()
    if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
        raise ValueError('Nonfinite model predictions')
    count = int(mask.sum())
    policy_loss = F.cross_entropy(logits[mask], actions[mask]) if count else logits.new_zeros(())
    value_loss = F.mse_loss(values, targets)
    loss = policy_loss + value_weight * value_loss
    with torch.no_grad():
        metrics = {
            'loss': loss.item(), 'policy_loss': policy_loss.item() if count else None,
            'value_loss': value_loss.item(), 'policy_sample_count': count,
            'value_sample_count': len(targets),
            'policy_top1_full_action': (logits[mask].argmax(1) == actions[mask]).float().mean().item() if count else None,
            'zero_value_baseline_mse': targets.square().mean().item(),
            'predicted_value_mean': values.mean().item(), 'target_value_mean': targets.mean().item(),
        }
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = math.sqrt(sum(float(p.grad.detach().norm()) ** 2
                                  for p in model.parameters() if p.grad is not None))
    if not math.isfinite(gradient_norm):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError('Nonfinite gradients')
    optimizer.step()
    metrics['gradient_norm'] = gradient_norm
    return metrics


@torch.inference_mode()
def evaluate_batches(model, batches, device='cpu'):
    """Aggregate by example counts, never by unweighted batch means.

    Move metrics here use all network actions. The separate search evaluator
    reports legal-conditioned human-move prediction on its shared sample.
    """
    was_training = model.training
    model.eval()
    policy_count = value_count = top1 = top5 = 0
    nll = squared_error = zero_error = absolute_error = predicted_sum = target_sum = 0.0
    try:
        for batch in batches:
            actions = batch['actions'].to(device)
            targets = batch['values'].to(device)
            mask = batch['policy_mask'].to(device=device, dtype=torch.bool)
            logits, values = model(batch['states'].to(device))
            values = values.flatten()
            if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
                raise ValueError('Nonfinite validation predictions')
            selected = int(mask.sum())
            if selected:
                selected_logits, selected_actions = logits[mask], actions[mask]
                nll += float(F.cross_entropy(selected_logits, selected_actions, reduction='sum'))
                top1 += int((selected_logits.argmax(1) == selected_actions).sum())
                top5 += int((selected_logits.topk(min(5, logits.shape[1]), dim=1).indices == selected_actions[:, None]).any(1).sum())
                policy_count += selected
            value_count += len(targets)
            squared_error += float((values - targets).square().sum())
            absolute_error += float((values - targets).abs().sum())
            zero_error += float(targets.square().sum())
            predicted_sum += float(values.sum())
            target_sum += float(targets.sum())
    finally:
        model.train(was_training)
    return {
        'policy_sample_count': policy_count, 'value_sample_count': value_count,
        'policy_nll_full_action': nll / policy_count if policy_count else None,
        'policy_top1_full_action': top1 / policy_count if policy_count else None,
        'policy_top5_full_action': top5 / policy_count if policy_count else None,
        'value_mse': squared_error / value_count if value_count else None,
        'value_mae': absolute_error / value_count if value_count else None,
        'zero_value_baseline_mse': zero_error / value_count if value_count else None,
        'predicted_value_mean': predicted_sum / value_count if value_count else None,
        'target_value_mean': target_sum / value_count if value_count else None,
    }


def train_supervised(dataset_dir, run_dir, *, epochs=1, batch_size=128, hidden=128,
                     learning_rate=.001, value_weight=1.0, seed=1, threads=4,
                     device='cpu', resume=None):
    """Run full streaming epochs and checkpoint at every completed epoch.

    Resume files contain optimizer/RNG state and an epoch boundary. A fresh
    output directory is required; an interrupted partial epoch can be replayed
    from its last complete checkpoint without pretending those steps survived.
    """
    if min(epochs, batch_size, hidden, threads) < 1:
        raise ValueError('Epochs, batch size, hidden width and threads must be positive')
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError('Invalid learning rate or value weight')
    dataset_dir, run_dir = Path(dataset_dir).resolve(), Path(run_dir).resolve()
    manifest = load_manifest(dataset_dir)
    manifest_hash = sha256(dataset_dir / 'manifest.json')
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Run directory must be empty or new')
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / 'checkpoints'
    checkpoints.mkdir()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    model = PolicyValueNet(hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = step = 0
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        state = saved['training_state']
        expected = {'dataset_manifest_sha256': manifest_hash, 'batch_size': batch_size,
                    'hidden': hidden, 'seed': seed, 'value_weight': value_weight,
                    'learning_rate': learning_rate, 'threads': threads, 'device': device}
        if any(state[key] != value for key, value in expected.items()):
            raise ValueError('Resume configuration or dataset does not match checkpoint')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        torch.set_rng_state(saved['torch_rng_state'])
        random.setstate(saved['python_rng_state'])
        if torch.device(device).type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng_state_all'])
        elif torch.device(device).type == 'mps':
            torch.mps.set_rng_state(saved['mps_rng_state'])
        start_epoch, step = state['completed_epoch'], state['step']
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {}
    for path in sorted((source_root / 'chess/scr').glob('*.py')):
        relative = path.relative_to(source_root)
        destination = run_dir / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_hashes[str(relative)] = sha256(path)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=source_root, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    config = {'dataset_dir': str(dataset_dir), 'dataset_manifest_sha256': manifest_hash,
              'run_dir': str(run_dir), 'epochs': epochs, 'start_epoch': start_epoch,
              'batch_size': batch_size, 'hidden': hidden, 'learning_rate': learning_rate,
              'value_weight': value_weight, 'seed': seed, 'threads': threads, 'device': device,
              'resume': str(resume) if resume else None, 'torch_version': str(torch.__version__),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'git_commit': commit, 'git_dirty': dirty, 'source_sha256': source_hashes,
              'started_at': datetime.now(timezone.utc).isoformat(),
              'policy_supervision': 'target player only; cross entropy over all actions',
              'checkpoint_selection': 'best validation policy NLL among initial model and completed epochs in this run',
              'value_supervision': 'observed PGN outcome from each side to move; both players',
              'value_limitations': 'human outcomes include resignation/timeouts and future mistakes; not minimax labels'}
    write_json(run_dir / 'config.json', config)
    write_json(run_dir / 'dataset_manifest.json', manifest)
    save_model(model, checkpoints / 'initial.pt')
    initial_validation = evaluate_batches(model, iter_batches(dataset_dir, 'validation', batch_size), device)
    if not initial_validation['policy_sample_count']:
        raise ValueError('Validation split has no target-player positions')
    # A worse epoch must not displace the starting model as the best candidate.
    best_nll = initial_validation['policy_nll_full_action']
    save_model(model, checkpoints / 'best.pt')
    history = [{'epoch': start_epoch, 'step': step, 'validation': initial_validation}]
    write_json(run_dir / 'validation.json', history)
    summary = {'status': 'running', 'completed_epoch': start_epoch, 'optimizer_steps': step,
               'initial_validation': initial_validation, 'epochs': [], 'checkpoints': []}
    write_json(run_dir / 'summary.json', summary)
    print(f'Initial validation: {initial_validation}', flush=True)
    started = time.monotonic()
    try:
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            epoch_started = time.monotonic()
            policy_count = value_count = 0
            policy_sum = value_sum = 0.0
            for batch in iter_batches(dataset_dir, 'train', batch_size, seed=seed + epoch, shuffle=True):
                update_started = time.monotonic()
                metrics = supervised_step(model, optimizer, batch, device, value_weight)
                step += 1
                metrics.update(step=step, epoch=epoch, duration_seconds=time.monotonic() - update_started)
                with (run_dir / 'updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + '\n')
                policy_count += metrics['policy_sample_count']
                value_count += metrics['value_sample_count']
                policy_sum += (metrics['policy_loss'] or 0) * metrics['policy_sample_count']
                value_sum += metrics['value_loss'] * metrics['value_sample_count']
                if step % 100 == 0:
                    print(f'Epoch {epoch}, step {step}: policy CE={policy_sum / max(1, policy_count):.3f}, '
                          f'value MSE={value_sum / max(1, value_count):.3f}, '
                          f'policy/value samples={policy_count}/{value_count}', flush=True)
            if not policy_count:
                raise ValueError('Training split has no target-player positions')
            validation = evaluate_batches(model, iter_batches(dataset_dir, 'validation', batch_size), device)
            history.append({'epoch': epoch, 'step': step, 'validation': validation})
            write_json(run_dir / 'validation.json', history)
            improved = validation['policy_nll_full_action'] < best_nll
            best_nll = min(best_nll, validation['policy_nll_full_action'])
            state = {key: config[key] for key in ('dataset_manifest_sha256', 'batch_size', 'hidden',
                                                 'seed', 'value_weight', 'learning_rate', 'threads', 'device')}
            state.update(completed_epoch=epoch, step=step, best_validation_nll=best_nll)
            checkpoint = checkpoints / f'epoch_{epoch}.pt'
            snapshot = {'format_version': 2, 'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'training_state': state,
                        'torch_rng_state': torch.get_rng_state(), 'python_rng_state': random.getstate()}
            if torch.device(device).type == 'cuda':
                snapshot['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
            elif torch.device(device).type == 'mps':
                snapshot['mps_rng_state'] = torch.mps.get_rng_state()
            torch.save(snapshot, checkpoint)
            if improved:
                save_model(model, checkpoints / 'best.pt')
            epoch_record = {'epoch': epoch, 'step': step, 'policy_samples': policy_count,
                            'value_samples': value_count, 'policy_loss': policy_sum / policy_count,
                            'value_loss': value_sum / value_count, 'validation': validation,
                            'duration_seconds': time.monotonic() - epoch_started}
            summary['epochs'].append(epoch_record)
            summary['checkpoints'].append(str(checkpoint))
            summary.update(completed_epoch=epoch, optimizer_steps=step, last_validation=validation,
                           duration_seconds=time.monotonic() - started)
            write_json(run_dir / 'summary.json', summary)
            print(f'Completed epoch {epoch}: {validation}', flush=True)
        save_model(model, checkpoints / 'final.pt')
        summary.update(status='completed', final_checkpoint=str(checkpoints / 'final.pt'),
                       best_checkpoint=str(checkpoints / 'best.pt') if (checkpoints / 'best.pt').exists() else None,
                       completed_at=datetime.now(timezone.utc).isoformat(),
                       duration_seconds=time.monotonic() - started)
        write_json(run_dir / 'summary.json', summary)
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       optimizer_steps=step, error=f'{type(exc).__name__}: {exc}',
                       duration_seconds=time.monotonic() - started)
        write_json(run_dir / 'summary.json', summary)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--value-weight', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--resume', type=Path)
    args = vars(parser.parse_args(argv))
    print(json.dumps(train_supervised(**args), indent=2))


if __name__ == '__main__':
    main()
