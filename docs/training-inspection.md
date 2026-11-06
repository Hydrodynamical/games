# Reading a training experiment

The training loop has three different objects to inspect: games producing targets, the network fitting those targets, and the resulting choices on fixed chess positions. Each gets a separate log. A model can improve at fitting its data while remaining a weak player.

## How did games end?

`games.jsonl` records `checkmate`, `stalemate`, `threefold_repetition`, `fifty_move_rule`, `insufficient_material`, or `move_limit`. A game that ends normally on its final allowed ply is recorded by its actual ending, not as a cutoff. Move-limit games have `truncated: true` and chess outcome `*`.

The numeric `result` is White's outcome value for completed games. It remains a legacy zero placeholder for move-limit games, but the runner now sets `value_target_available: false` and excludes those rows from value loss by default (`--cutoff-value-policy exclude`). Their policy targets remain in replay and train normally. Genuine rule-defined draws remain eligible zero-valued examples. `--cutoff-value-policy draw` reproduces the original assumption that unfinished games have value zero; missing policy flags in historical logs mean that legacy behavior.

Value loss is the mean squared error over eligible rows, normalized by their count. A batch with no eligible outcomes has zero value-loss contribution and unavailable value metrics, rather than a reported perfect MSE. The value-head gradients remain absent for that step, including when Adam has existing momentum. The shared convolutional trunk still learns from policy examples, so value predictions can change indirectly. Normalizing over eligible rows preserves the completed examples' contribution even when many batch rows are excluded; this also changes their relative weight compared with the old full-batch average.

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
- **Value loss / MAE:** squared/absolute error against outcome labels on the value-eligible subset. The legacy draw policy includes cutoff zeros; the default excludes them. Do not compare these training losses across policies as if they used the same target population.
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

To compare the legacy draw-label control with the value mask, start two fresh runs using identical settings and seeds, changing only `--cutoff-value-policy draw` to `exclude` and the output directory. Then run:

```bash
PYTHONPATH=chess .venv/bin/python -m scr.compare_training \
  runs/draw-control runs/value-mask --output /tmp/chess-comparison.json
```

Both full run folders need their initial and final checkpoints. The comparator records configuration differences, checks exact initial tensors, initial probes, pre-update game histories and first policy loss, and reports whether these checks passed. It counts policy/value samples separately and evaluates the same six known-outcome positions with the initialization and both final checkpoints. The set balances two mate-in-one wins, two already-checkmated losses, and two bare-kings draws; the constant-zero MSE is 2/3. These synthetic positions are evaluation-only, include terminal states normally handled directly by search, and are not a playing-strength benchmark.

After the intervention, search and replay can diverge even with the same seed. Different eligible populations make training value losses unsuitable for a direct accuracy comparison; policy losses also use each run's own sampled search targets. Read the [recorded comparison](../experiments/2026-11-06-cutoff-value-mask-01/README.md) for the first matched result and its limits.
