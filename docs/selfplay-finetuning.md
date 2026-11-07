# From human imitation to MCTS self-play

A supervised checkpoint can initialize the existing self-play runner. This starts a new training stage: the network weights are reused, while Adam and the replay buffer start fresh. Policy targets become normalized MCTS visit counts for both players. Value targets become observed self-play outcomes from the player-to-move perspective.

The original supervised checkpoint stays unchanged. Record its content hash and fix the training budget, evaluation games, and search settings before inspecting the candidate's results. A decreasing replay loss alone does not establish better value prediction or stronger play.

## Freeze an independent value benchmark

Generate a separate fixed-budget collection of games with the supervised model. The benchmark uses an independent seed and never enters training replay:

```bash
PYTHONPATH=chess .venv/bin/python -u -m scr.value_benchmark \
  --checkpoint runs/2026-11-06-hikaru-supervised-02/checkpoints/best.pt \
  --output-dir runs/2026-11-06-hikaru-value-benchmark-01 \
  --games 20 --sims 64 --max-plies 400 --temperature 1 \
  --seed 20261107 --threads 4
```

The manifest records the source model, settings, source hashes, and game-file hash. Full histories preserve draw and repetition state. Every generated game is retained, including cutoffs; games are not selectively replaced to obtain favorable outcomes.

Value evaluation uses all pre-move positions from naturally completed games and compares both models on exactly the same inputs and outcomes. Cutoff games have unknown outcomes and are excluded from evaluation labels. Report decisive and drawn games separately, the constant-zero baseline, and uncertainty by game rather than treating adjacent positions as independent.

These targets describe outcomes under the frozen supervised model's exploratory self-play policy. They are neither perfect-play evaluations nor outcomes under the candidate's new policy. Common opening positions can occur in both training and benchmark games; entire benchmark histories remain outside replay.

## Train a fixed self-play budget

```bash
PYTHONPATH=chess .venv/bin/python -u -m scr.main \
  --resume runs/2026-11-06-hikaru-supervised-02/checkpoints/best.pt \
  --run-dir runs/2026-11-06-hikaru-selfplay-01 \
  --iterations 100 --sims 64 --max-moves 400 --temperature 1 \
  --batch-size 64 --min-buffer 256 --replay-capacity 20000 --updates 8 \
  --learning-rate 0.0001 --seed 2 --device cpu --threads 4 \
  --cutoff-value-policy draw --save-every 10 --trace-every 10 \
  --probe-every 10 --probe-sims 64 --report-every 10
```

The pilot keeps the existing cutoff-zero training policy. Those labels can bias the value head toward zero, so logs expose their prevalence separately from genuine draws. This differs deliberately from value evaluation, where an unknown outcome supplies no accuracy target. No new value mask is enabled for this experiment.

Root exploration noise and temperature one diversify training games. Search trees are reused within a game and discarded before any weight update. The maximum game length is 400 plies (200 full moves). The replay buffer is bounded at 20,000 positions. There are up to eight updates per game after warmup; the actual update count depends on when the buffer reaches 256 positions.

The fixed final candidate is `checkpoints/policy_value_net_100.pt`. Intermediate checkpoints support retrospective learning curves, not evaluation-based checkpoint selection. These self-play checkpoints contain weights only; resuming them again resets optimizer and replay state.

## Compare both models with the same search budget

```bash
PYTHONPATH=chess .venv/bin/python -u -m scr.evaluate_models \
  runs/2026-11-06-hikaru-supervised-02/checkpoints/best.pt \
  runs/2026-11-06-hikaru-selfplay-01/checkpoints/policy_value_net_100.pt \
  runs/2026-11-06-hikaru-value-benchmark-01 \
  runs/2026-11-06-hikaru-selfplay-01/comparison \
  --match-pairs 10 --sims 64 --max-plies 400 --seed 1 --threads 4
```

Both sides use MCTS with the same simulation budget, no exploration noise, deterministic move selection, and a fresh tree per decision. Each fixed opening is played with model colors swapped. Wins, losses, genuine draws, and unresolved cutoffs are separate. Completed-only scores omit cutoffs explicitly; bounds over all games show the possible impact of unknown results. Equal simulations do not imply identical wall-clock time.

The comparison also scores the frozen value benchmark with both models, preserving per-position predictions and per-game metrics. Full human test-split metrics can be recorded separately to inspect retention of the original task. Neither comparison is fed back into this pilot's optimization or model selection. Twenty matches and one training seed provide diagnostic evidence, not an Elo estimate.

The existing `report.html` inspects training losses, endings, probes, and traced games. Generate the comparison report after writing `comparison/`, `value_curve.json`, and `human_retention.json`:

```bash
PYTHONPATH=chess .venv/bin/python -m scr.selfplay_report \
  --run-dir runs/2026-11-06-hikaru-selfplay-01
```

Its default output is `selfplay_report.html`; the original training report remains available. All full runs, benchmark games, and weights stay local under ignored `runs/`. Compact evidence and reproduction instructions belong in the experiment archive.
