# Reading a training experiment

The training loop has three different objects to inspect: games producing targets, the network fitting those targets, and the resulting choices on fixed chess positions. Each gets a separate log. A model can improve at fitting its data while remaining a weak player.

## How did games end?

`games.jsonl` records `checkmate`, `stalemate`, `threefold_repetition`, `fifty_move_rule`, `insufficient_material`, or `move_limit`. A game that ends normally on its final allowed ply is recorded by its actual ending, not as a cutoff. Move-limit games have `truncated: true` and chess outcome `*`.

The numeric `result` is White's outcome value for completed games. Move-limit games receive zero-valued training labels by default (`--cutoff-value-policy draw`), with `value_target_available: true`; their policy targets also train normally. This is a training assumption, while their recorded chess outcome remains unfinished. Genuine rule-defined draws remain eligible zero-valued examples. Missing policy flags in historical logs also mean draw labeling. The experimental `--cutoff-value-policy exclude` remains available to reproduce the value-mask comparison: it keeps cutoff positions for policy training but sets `value_target_available: false` and excludes their value loss.

Value loss is the mean squared error over eligible rows, normalized by their count. Under the default draw policy, every row is eligible. With the experimental exclusion policy, a batch containing only cutoffs has zero value-loss contribution and unavailable value metrics, rather than a reported perfect MSE. The value-head gradients remain absent for that step, including when Adam has existing momentum. The shared convolutional trunk still learns from policy examples, so value predictions can change indirectly. Normalizing over eligible rows preserves the completed examples' contribution even when many batch rows are excluded; this also changes their relative weight compared with the default full-batch average.

The summary reports cutoff-origin positions, value-eligible positions, and excluded-value positions separately. The legacy `cutoff_target_positions`/`cutoff_target_fraction` fields describe the origin of policy/replay examples, not how many received a value label. Update logs contain policy/value sample counts, eligible-only outcome fractions, and whole-policy-batch origin fractions; compare like denominators.

## What does one decision show?

Each detailed trace describes the position **before** the recorded move. Its `value_prediction` is the raw network prediction from the side-to-move's perspective: positive favors that player. It is not a centipawn score or a calibrated probability of winning.

`prior` is the network's distribution conditional on legal moves, before root exploration noise. `search_probability` is the normalized MCTS visit distribution after search, including retained subtree visits. The actual sampled move may differ from either distribution's top move because self-play uses temperature sampling. `visits` and `q` belong to the root's action; `q` is the search mean value from the current player's perspective. `sims` counts new simulations for that ply, rather than all retained visits.

Policy and search entropy measure how spread out the legal move distributions are, using natural logarithms. A flat distribution over N moves has entropy log(N). Lower entropy means more concentrated choices, which may be confidently wrong. Trace `policy_search_kl` measures how far the search distribution moved from the clean legal policy. A large change indicates search supplied a different target, not necessarily a better chess move.

## What does an optimizer update show?

All losses/predictions are evaluated **before** that update on its sampled batch. Values cannot be compared as if the same examples occurred at every step: the replay buffer and minibatches change.

- **Policy loss:** cross-entropy against the search visit target, over all 4,480 possible action outputs. It includes learning to suppress illegal actions.
- **Policy target entropy:** the target's inherent uncertainty. Cross-entropy cannot become zero when the target is a spread-out distribution.
- **Policy KL:** policy loss minus target entropy. This measures mismatch beyond that uncertainty. Unlike the game-trace KL, it uses the full action distribution, not the model conditioned on legal moves.
- **Top-one agreement:** the network's highest-scored action equals the target's highest-probability action. Ties are broken by action index; do not over-interpret this when visits are nearly uniform.
- **Value loss / MAE:** squared/absolute error against outcome labels on the value-eligible subset. The default draw policy includes cutoff zeros; experimental exclusion omits them. Do not compare these training losses across policies as if they used the same target population.
- **Constant-zero value baseline:** the MSE from always predicting zero on the same value-eligible rows. With -1/0/+1 labels this equals the fraction of decisive targets among those rows. Compare value loss against this baseline to separate fitting useful position values from a changing mix of zero labels.
- **Value means and standard deviations:** use only value-eligible rows. These are null when none exist. They can reveal a network converging toward zero when labels are mostly zero; low value loss on such batches does not demonstrate understanding of winning positions.
- **Gradient norm:** size of the gradient before the optimizer step. This is diagnostic only; gradients are not clipped or rescaled by the logger.

The metrics themselves do not change the optimizer. The cutoff-value policy selects the value supervision described above. The learning rate remains configurable, defaulting to 0.001 with Adam. Default training still runs two optimizer steps per game after replay warmup.

## What do the fixed probes establish?

Six positions cover the initial board, hanging-queen captures for both colors, mate-in-one for both colors, and a known kings-only draw. The report shows raw network values, preferred moves, expected-move probability, and MCTS decisions separately. MCTS can solve an immediate mate even when its network policy is poor, so a search pass is not a learned-policy pass.

Probes never enter the replay buffer. They use no exploration noise or move sampling, and preserve random-number generator states and model modes. They make changes easier to interpret on a stable set of inputs, but they are a tiny diagnostic set, not a general-strength benchmark or an Elo estimate.

## Reproduce and compare

Each run records its parameters, environment, Git commit/dirty status, exact source copies and hashes. Initial and periodic network checkpoints are preserved. Logs use strict JSON (nonfinite numbers are rejected). Existing run folders and checkpoint filenames are protected against overwrite. Optional traces add an extra network evaluation per recorded move and consume disk space; select `--trace-every` to control their frequency.

Current checkpoints still contain network weights only. `--resume` starts a fresh optimizer and replay buffer in a new run; it is not an exact training continuation. Randomness is seeded, but exact numerical reproduction across devices, library versions, or threading configurations is not guaranteed. Record those settings when comparing experiments.

To reproduce the comparison between the default draw-label policy and the experimental value mask, start two fresh runs using identical settings and seeds, changing only `--cutoff-value-policy draw` to `exclude` and the output directory. Then run:

```bash
PYTHONPATH=chess .venv/bin/python -m scr.compare_training \
  runs/draw-control runs/value-mask --output /tmp/chess-comparison.json
```

Both full run folders need their initial and final checkpoints. The comparator records configuration differences, checks exact initial tensors, initial probes, pre-update game histories and first policy loss, and reports whether these checks passed. It counts policy/value samples separately and evaluates the same six known-outcome positions with the initialization and both final checkpoints. The set balances two mate-in-one wins, two already-checkmated losses, and two bare-kings draws; the constant-zero MSE is 2/3. These synthetic positions are evaluation-only, include terminal states normally handled directly by search, and are not a playing-strength benchmark.

After the intervention, search and replay can diverge even with the same seed. Different eligible populations make training value losses unsuitable for a direct accuracy comparison; policy losses also use each run's own sampled search targets. Read the [recorded comparison](../experiments/2026-11-06-cutoff-value-mask-01/README.md) for the first matched result and its limits.

## Planning 5,000 games locally

On the Apple M4 / 16 GiB Mac measured on November 6, 2026, the draw-label pilot took 149 seconds for 12 games, including 99.4 seconds of self-play. Straight-line wall-time extrapolation is 17.2 hours for 5,000 games. With the same CPU/four-thread configuration, 64 simulations per ply, 200-ply cap, 128 channels, batch size 64, and two updates per game, budget roughly **12–24 hours** with sparse reporting. Increasing search to 100 simulations and batch size to 256 suggests roughly **18–30 hours**, but that configuration has not had a full self-play benchmark. Game lengths, CPU contention, sustained clock speed, and filesystem overhead can change these estimates.

Replay is capped at 20,000 positions rather than retaining all 5,000 games' tensors. Its float32 state/policy payload is 460.8 MB (439 MiB); model weights, gradients, and Adam moments add about 153 MiB. A short capacity check with a full replay and five CPU updates peaked at about 0.8–0.9 GiB resident memory for batches 64 and 256. This is not a measured end-to-end peak: allow **2–4 GiB of RAM headroom** for self-play, runtime allocations, and report generation. The 16 GiB machine has sufficient capacity with this bounded replay and sampled traces.

Disk cadence matters. One checkpoint is about 38.3 MiB. Saving every five games creates 1,001 files, about **37.4 GiB**; only about 20 GiB was free when checked. Saving every 100 games creates 51 files, about **1.9 GiB**. With detailed traces every 50 games and probes/reports every 100, the observed logs project to about **2.1 GB total storage**. Every game's ending and UCI history is still retained. Detailed traces for every game and a report after every game are inappropriate at this scale: the report rereads and embeds all accumulated logs on each refresh.

A proposed CPU configuration preserving the pilot's learning/search settings is:

```bash
.venv/bin/python -u chess/scr/main.py \
  --iterations 5000 --sims 64 --max-moves 200 --hidden 128 \
  --batch-size 64 --min-buffer 256 --replay-capacity 20000 --updates 2 \
  --seed 1 --device cpu --threads 4 --cutoff-value-policy draw \
  --run-dir runs/long-5000-01 --save-every 100 --trace-every 50 \
  --probe-every 100 --probe-sims 80 --report-every 100
```

This is a capacity plan, not a completed long run or a guarantee of stronger play. At two updates per game, 5,000 games produce roughly 10,000 optimizer steps; inspect fixed probes throughout. Before relying on interruption recovery, note that current checkpoints save weights only: exact continuation would also require saving optimizer, replay, and random-number-generator state.
