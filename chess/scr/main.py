"""Inspectable neural MCTS training with game, update, and fixed-probe logs."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import time
import uuid

import torch

try:
    from .chess_engine import GameState
    from .rl_interface import PolicyValueNet, load_model, save_model, select_device
    from .self_play import play_self_game
    from .train import train_step
    from .experiment import RunLogger
    from .diagnostics import evaluate_probes
    from .report import generate_report
except ImportError:
    from chess_engine import GameState
    from rl_interface import PolicyValueNet, load_model, save_model, select_device
    from self_play import play_self_game
    from train import train_step
    from experiment import RunLogger
    from diagnostics import evaluate_probes
    from report import generate_report


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def replay_summary(replay, cutoff_value_policy="draw"):
    counts = Counter("cutoff" if item[3] else "draw" if item[2] == 0 else "decisive" for item in replay)
    excluded = counts["cutoff"] if cutoff_value_policy == "exclude" else 0
    return {"positions": len(replay), "cutoff_positions": counts["cutoff"],
            "draw_positions": counts["draw"], "decisive_positions": counts["decisive"],
            "policy_target_positions": len(replay), "value_target_positions": len(replay) - excluded,
            "excluded_value_positions": excluded}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=positive_int, default=1000, help="Self-play games to generate")
    parser.add_argument("--sims", type=positive_int, default=100, help="MCTS simulations per ply")
    parser.add_argument("--max-moves", type=positive_int, default=200, help="Maximum plies per game (200 = 100 full moves)")
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--min-buffer", type=positive_int, default=2000, help="Positions before the first training step")
    parser.add_argument("--replay-capacity", type=positive_int, default=20000)
    parser.add_argument("--updates", type=positive_int, default=2, help="Training steps per game after warmup")
    parser.add_argument("--run-dir", type=Path, help="New/empty output folder; defaults to runs/<UTC timestamp>-<id>")
    parser.add_argument("--checkpoint-dir", type=Path, help="Defaults to <run-dir>/checkpoints")
    parser.add_argument("--save-every", type=positive_int, default=5)
    parser.add_argument("--resume", type=Path, help="Load network weights; optimizer/replay start fresh")
    parser.add_argument("--hidden", type=positive_int, default=128, help="Convolution channels for a new model")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--cutoff-value-policy", choices=("draw", "exclude"), default="draw",
                        help="Keep cutoffs for policy and zero-valued outcomes (default); exclude reproduces the value-mask experiment")
    parser.add_argument("--trace-every", type=nonnegative_int, default=10,
                        help="Trace first game and every Nth game; 1 traces all, 0 disables")
    parser.add_argument("--probe-every", type=nonnegative_int, default=5,
                        help="Fixed probes at start/end and every N games; 0 disables")
    parser.add_argument("--probe-sims", type=positive_int, default=80)
    parser.add_argument("--report-every", type=positive_int, default=5)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.min_buffer > args.replay_capacity:
        parser.error("--min-buffer must not exceed --replay-capacity")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    source_root = Path(__file__).resolve().parents[2]
    if args.run_dir is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        args.run_dir = source_root / "runs" / name
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error(f"--run-dir is not empty: {args.run_dir}; choose a new directory")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.run_dir / "checkpoints"
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else select_device()
    model = load_model(args.resume, device) if args.resume else PolicyValueNet(hidden=args.hidden).to(device)
    initialization_hash = None
    if args.resume:
        digest = hashlib.sha256()
        with args.resume.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        initialization_hash = digest.hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), hidden=model.trunk[0].out_channels,
                  parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                  dirichlet_alpha=0.3, dirichlet_fraction=0.25, c_puct=1.5, neural_batch_size=16,
                  initialization="checkpoint_weights" if args.resume else "random",
                  initialization_checkpoint_sha256=initialization_hash,
                  optimizer_initialization="fresh Adam", replay_initialization="empty",
                  policy_target="normalized MCTS visit counts for both players",
                  value_target="self-play outcome from side-to-move perspective; cutoff policy recorded separately")
    logger = RunLogger(args.run_dir, config, torch_version=torch.__version__, source_root=source_root)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    replay = deque(maxlen=args.replay_capacity)
    step = 0
    print(f"Run directory: {args.run_dir}", flush=True)
    print(f"Training on {device}; {args.iterations} games, {args.sims} simulations/ply", flush=True)
    print("Move-limit positions train policy only; their value loss is excluded."
          if args.cutoff_value_policy == "exclude"
          else "Move-limit positions train both policy and zero-valued outcomes (default draw-label policy).", flush=True)

    def replay_state():
        return replay_summary(replay, args.cutoff_value_policy)

    def checkpoint(game):
        path = args.checkpoint_dir / f"policy_value_net_{game}.pt"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
        save_model(model, path)
        logger.checkpoint(path, game, step)
        print(f"Saved {path}", flush=True)

    def probes(game):
        started = time.monotonic()
        records = evaluate_probes(model, num_sims=args.probe_sims)
        logger.probes({"game": game, "step": step, "positions": records,
                       "duration_seconds": time.monotonic() - started})
        scored = [record for record in records if record["passed"] is not None]
        print(f"Fixed tactical probes: {sum(record['passed'] for record in scored)}/{len(scored)} "
              "(diagnostics, not an Elo estimate)", flush=True)

    try:
        checkpoint(0)
        if args.probe_every:
            probes(0)
        generate_report(args.run_dir)
        for iteration in range(1, args.iterations + 1):
            trace = bool(args.trace_every and (iteration == 1 or iteration % args.trace_every == 0))
            game_data, result, info = play_self_game(
                GameState(), model, num_mcts_sims=args.sims, temperature=args.temperature,
                max_moves=args.max_moves, show_progress=args.progress,
                return_info=True, record_trace=trace,
            )
            info["value_target_available"] = not info["truncated"] or args.cutoff_value_policy == "draw"
            for state, policy, player in game_data:
                value = result if player == "w" else -result
                replay.append((state.detach().cpu(), policy.detach().cpu(), float(value), info["truncated"]))
            logger.game(dict(info, game=iteration, step_before_updates=step, replay_size=len(replay)))
            summary = logger.refresh(replay=replay_state())
            print(f"Game {iteration}/{args.iterations}: {info['plies']} plies; {info['termination']}; "
                  f"outcome={info['outcome']}; cutoffs={summary['move_limit_games']}/{iteration}; "
                  f"replay={len(replay)}", flush=True)
            if len(replay) >= args.min_buffer:
                for _ in range(args.updates):
                    records = random.sample(list(replay), min(args.batch_size, len(replay)))
                    batch = [(state, policy, value) for state, policy, value, _ in records]
                    value_mask = [not record[3] or args.cutoff_value_policy == "draw" for record in records]
                    eligible = [record for record, selected in zip(records, value_mask) if selected]
                    started = time.monotonic()
                    stats = train_step(model, optimizer, batch, device, value_mask=value_mask)
                    step += 1
                    logger.update(dict(stats, game=iteration, step=step, batch_size=len(batch),
                                       buffer_size=len(replay), duration_seconds=time.monotonic() - started,
                                       cutoff_target_fraction=sum(record[3] for record in records) / len(records),
                                       excluded_value_fraction=1 - len(eligible) / len(records),
                                       cutoff_value_target_fraction=(sum(record[3] for record in eligible) / len(eligible)
                                                                     if eligible else None),
                                       value_target_win_fraction=(sum(record[2] > 0 for record in eligible) / len(eligible)
                                                                  if eligible else None),
                                       value_target_loss_fraction=(sum(record[2] < 0 for record in eligible) / len(eligible)
                                                                   if eligible else None),
                                       value_target_draw_fraction=(sum(record[2] == 0 for record in eligible) / len(eligible)
                                                                   if eligible else None),
                                       target_win_fraction=sum(record[2] > 0 for record in records) / len(records),
                                       target_loss_fraction=sum(record[2] < 0 for record in records) / len(records),
                                       actual_draw_target_fraction=sum(record[2] == 0 and not record[3]
                                                                       for record in records) / len(records)))
                print(f"Step {step}: policy loss={stats['policy_loss']:.4f}; "
                      f"value loss={stats['value_loss']:.4f}; policy KL={stats['policy_kl']:.4f}; "
                      f"value samples={stats['value_sample_count']}/{stats['policy_sample_count']}; "
                      f"gradient norm={stats['gradient_norm']:.4f}", flush=True)
            else:
                print(f"Warmup: {len(replay)}/{args.min_buffer} positions", flush=True)
            if args.probe_every and (iteration % args.probe_every == 0 or iteration == args.iterations):
                probes(iteration)
            if iteration % args.save_every == 0 or iteration == args.iterations:
                checkpoint(iteration)
            logger.refresh(replay=replay_state())
            if iteration % args.report_every == 0:
                generate_report(args.run_dir)
        summary = logger.refresh("completed", replay=replay_state())
        report = generate_report(args.run_dir)
        print(f"Completed: {summary['termination_counts']}; {step} updates. Report: {report}", flush=True)
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        logger.refresh(status, error=f"{type(exc).__name__}: {exc}", replay=replay_state())
        try:
            generate_report(args.run_dir)
        except Exception:
            pass  # Preserve the original failure; JSON logs remain inspectable.
        raise
    return model


if __name__ == "__main__":
    main()
