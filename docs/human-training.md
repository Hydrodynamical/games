# Human-game pretraining and MCTS comparison

Human games provide move and outcome targets before the engine generates its own self-play data. The existing 20-plane, 4,480-action policy/value network can consume these targets and immediately supply priors and leaf values to MCTS. This pipeline adds data ingestion, supervised optimization, held-out imitation measurements, and paired search comparisons; it does not change the self-play draw-label default.

## Download a reproducible dataset

Install the optional PGN/training dependencies in the existing Python 3.12 environment:

```bash
.venv/bin/python -m pip install -r requirements/human.txt
.venv/bin/python -u chess/scr/download_games.py \
  --username Hikaru --limit 5000 --output-dir data/hikaru-blitz-5000
```

The downloader uses Chess.com's [documented public API](https://www.chess.com/news/view/published-data-api). It requests the account profile and archive list, then fetches monthly JSON archives sequentially, newest first. Monthly JSON includes both PGNs and explicit speed/rules metadata. Selection requires rated standard chess, `time_class: blitz`, the requested account on exactly one side, a known result, and matching PGN/API players and outcomes. Wins, losses, draws, resignations, and clock endings are retained. Games are deduplicated by public game ID, and the newest requested number is written in chronological order.

Raw responses and their source URLs, dates, hashes, and HTTP cache validators are saved under `cache/`. `--offline` rebuilds a selection from the existing cache; `--refresh` revalidates responses. Requests have bounded retries/timeouts. Insufficient eligible games produce an explicit failed manifest and partial data rather than an apparently successful smaller dataset. A failed refresh preserves a previous completed selection.

The raw `games.jsonl` preserves PGNs and metadata; `manifest.json` records selection rules, verified profile, source hashes, date range, counts, and status. `data/` is excluded from Git. A completed cached download is a snapshot; getting newer games requires explicit refresh.

## Validate, split, and encode

```bash
PYTHONPATH=chess .venv/bin/python -u -m scr.human_data \
  --input data/hikaru-blitz-5000/games.jsonl \
  --output-dir data/hikaru-blitz-5000/prepared --shard-size 8192
```

PGNs are parsed with python-chess and every move is replayed through the custom engine. States are compared with the reference board and encoded before each move, preserving castling, en passant, clock, and repetition history. Underpromotions use the engine's existing action encoding. Resignation and agreed-draw results remain valid even when the final board is not terminal. Unknown, inconsistent, empty, or malformed games are recorded as rejected; accepted/rejected counts are explicit.

Training, validation, and test splits target 80/10/10 chronologically, keeping whole UTC dates together. Actual counts can differ from exact percentages. Game IDs and full-game fingerprints are deduplicated before splitting; adjacent positions from a game never appear in different splits. Common opening positions can still recur across different games, so opening and later-position imitation scores are also reported separately. This is a chronological game holdout, not a position-disjoint or opponent-disjoint benchmark.

Prepared records retain initial FENs, full UCI histories, player color, result, and provenance. Arrays are compressed into bounded shards: binary planes are uint8, the halfmove-clock plane is stored on a 0–100 scale, and repetition on a 0–3 scale. Action indices are compact integers, not dense 4,480-entry one-hot vectors. Only a minibatch is expanded to float32. Shard and game-file hashes are checked when read.

## Supervised optimization

```bash
PYTHONPATH=chess .venv/bin/python -u -m scr.supervised \
  --dataset-dir data/hikaru-blitz-5000/prepared \
  --run-dir runs/hikaru-supervised-01 \
  --epochs 1 --batch-size 128 --hidden 128 --learning-rate 0.001 \
  --value-weight 1 --seed 1 --threads 4 --device cpu
```

The policy loss is cross-entropy against **Hikaru's moves only**. Opponent moves reconstruct the board and supply value examples, but do not become imitation targets. The value loss is MSE against the recorded result from the player-to-move perspective, using both players' positions. Its target is observed human outcome, not perfect-play evaluation; ratings, future mistakes, time pressure, and timeouts introduce noise. The current network does not encode player identity or remaining clock time. Training on one famous player's games is a starting point, not evidence that the whole playing style has been captured.

Each update records pre-update loss, sample counts, full-action policy accuracy, value statistics, and gradient norm. Validation aggregates by example count rather than averaging unequal minibatches. Training/validation policy metrics use the **full action space**, including illegal outputs; the separate MCTS evaluation conditions policy probabilities on legal moves. These are distinct denominators and should not be compared directly.

Run folders are protected against overwrite and retain configuration, dataset manifest hash, source snapshots/hashes, updates, validation measurements, and initial/final/best weights. `best.pt` means the lowest validation policy NLL among the starting model and completed epochs **in that run**, including a continuation run. It does not automatically retrieve a better model from a previous run.

Every `epoch_N.pt` additionally contains Adam state, RNG states, and an epoch boundary. Resume into a fresh directory with the same dataset and training settings:

```bash
PYTHONPATH=chess .venv/bin/python -u -m scr.supervised \
  --dataset-dir data/hikaru-blitz-5000/prepared \
  --run-dir runs/hikaru-supervised-continuation \
  --resume runs/hikaru-supervised-01/checkpoints/epoch_1.pt \
  --epochs 1 --batch-size 128 --hidden 128 --learning-rate 0.001 \
  --value-weight 1 --seed 1 --threads 4 --device cpu
```

Here `--epochs` is the number of additional complete epochs. Interrupted partial epochs restart from the last completed epoch checkpoint. CPU continuation is tested against uninterrupted training for identical weights. Accelerator RNG states are saved, but exact accelerator numerical reproduction has not been established. The older self-play runner's `--resume` remains weights-only.

## Compare the same model with and without search

```bash
PYTHONPATH=chess .venv/bin/python -u -m scr.evaluate_human \
  runs/hikaru-supervised-01/checkpoints/best.pt \
  data/hikaru-blitz-5000/prepared runs/hikaru-supervised-01/eval-final \
  --max-positions 256 --sims 64 --seed 1 --threads 4 \
  --match-pairs 6 --max-plies 160
```

The evaluator samples playable Hikaru positions uniformly from the test split using a fixed seed and full replay history. Positions the custom engine automatically treats as drawn are explicitly excluded even if the human PGN continued. The two modes share exactly the same checkpoint and board positions:

- **Raw policy:** choose the most probable legal move from one network evaluation.
- **MCTS:** use that policy and value network for 64 simulations, with no root noise, no move sampling, and a fresh tree per decision.

Outputs include top-one/top-five human-move agreement, raw legal-conditioned negative log likelihood, value error, changed decisions, and time per decision, with opening/later breakdowns. Search visit probabilities can be zero; the report does not invent smoothed search NLLs. Matching Hikaru's recorded move measures imitation, not correctness.

Both modes also run on fixed tactical probes. Six two-ply openings each produce two head-to-head games with colors swapped. The cap is 160 additional plies after the opening. Wins, genuine draws, and unresolved cutoffs are separate; scores over completed games omit cutoffs explicitly. These are twelve diagnostic games, not an Elo estimate. MCTS uses more computation, so this is not an equal-time match.

Evaluate `checkpoints/initial.pt` into `eval-initial` with the same sample seed and `--match-pairs 0` to measure the effect of pretraining. Select the checkpoint using validation before examining test results. Never feed evaluation examples back into the training set.

An offline report can be regenerated after the evaluations:

```bash
PYTHONPATH=chess .venv/bin/python -m scr.human_report --run-dir runs/hikaru-supervised-01
```

The full local run and datasets are ignored by Git. Tracked experiment summaries contain compact evidence and reproduction instructions, while raw downloads and trained weights remain local.
