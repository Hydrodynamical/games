# Reading a training experiment

The training loop has three different objects to inspect: games producing targets, the network fitting those targets, and the resulting choices on fixed chess positions. Each gets a separate log. A model can improve at fitting its data while remaining a weak player.

## How did games end?

`games.jsonl` records `checkmate`, `stalemate`, `threefold_repetition`, `fifty_move_rule`, `insufficient_material`, or `move_limit`. A game that ends normally on its final allowed ply is recorded by its actual ending, not as a cutoff. Move-limit games have `truncated: true` and chess outcome `*`.

The numeric `result` is White's value-training label. It remains zero for move-limit games, matching the previous training policy. This is an assumption made to train on unfinished games, not a discovered draw. The summary reports both the fraction of games cut off and the fraction of generated positions whose targets came from cutoffs. Updates also record that fraction in the actual sampled batch, together with genuine draw/win/loss fractions.

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
- **Value loss / MAE:** squared/absolute error against final outcome labels, including cutoff zeros under the current policy.
- **Constant-zero value baseline:** the MSE from always predicting zero on the same batch. With -1/0/+1 labels this equals the fraction of decisive targets. Compare value loss against this baseline to separate fitting useful position values from a changing mix of zero labels.
- **Value means and standard deviations:** reveal a network converging toward zero when almost all labels are zero. Low value loss on such batches does not demonstrate understanding of winning positions.
- **Gradient norm:** size of the gradient before the optimizer step. This is diagnostic only; gradients are not clipped or rescaled by the logger.

No optimizer or reward change is introduced by these metrics. The learning rate remains configurable, defaulting to 0.001 with Adam. Default training still runs two optimizer steps per game after replay warmup.

## What do the fixed probes establish?

Six positions cover the initial board, hanging-queen captures for both colors, mate-in-one for both colors, and a known kings-only draw. The report shows raw network values, preferred moves, expected-move probability, and MCTS decisions separately. MCTS can solve an immediate mate even when its network policy is poor, so a search pass is not a learned-policy pass.

Probes never enter the replay buffer. They use no exploration noise or move sampling, and preserve random-number generator states and model modes. They make changes easier to interpret on a stable set of inputs, but they are a tiny diagnostic set, not a general-strength benchmark or an Elo estimate.

## Reproduce and compare

Each run records its parameters, environment, Git commit/dirty status, exact source copies and hashes. Initial and periodic network checkpoints are preserved. Logs use strict JSON (nonfinite numbers are rejected). Existing run folders and checkpoint filenames are protected against overwrite. Optional traces add an extra network evaluation per recorded move and consume disk space; select `--trace-every` to control their frequency.

Current checkpoints still contain network weights only. `--resume` starts a fresh optimizer and replay buffer in a new run; it is not an exact training continuation. Randomness is seeded, but exact numerical reproduction across devices, library versions, or threading configurations is not guaranteed. Record those settings when comparing experiments.
