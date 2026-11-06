# Hydrodynamical Chess

A custom Python chess rules engine with playable Monte Carlo tree search (MCTS), a Tk board, a command-line interface, and a basic UCI adapter. The default search uses a handcrafted evaluator and needs no trained model or third-party packages. An optional PyTorch policy/value network supports batched search and self-play training.

This is an experimental engine with no established Elo rating. Rules correctness and a reproducible search/training pipeline are the current focus.

## Play now

Use Python 3.12 or newer. Run commands from the repository root:

```bash
python3 chess/cli.py play --you w --sims 400
```

Enter moves such as `e2e4`, `e1g1` (castling), or `a7a8n` (promotion). `moves` lists legal moves, `undo` takes back your turn and the engine reply, `fen` prints the position, and `quit` exits. Use `--you b` to play Black. `--time 1` additionally limits each search to approximately one second; the first simulation or time limit reached ends the search.

To use the graphical board:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/gui.txt
.venv/bin/python chess/UI_with_AI.py --you w --sims 400
```

The GUI also requires Tk support in your Python installation. Left-click source and destination squares, right-click to highlight legal moves, and press `u` to undo. Promotion prompts for Q/R/B/N. The engine searches in a worker thread, keeping the board responsive. The older two-player board is still available at `chess/main.py`.

## Inspect positions

```bash
python3 chess/cli.py analyze --sims 1000
python3 chess/cli.py analyze --fen '7k/8/5KQ1/8/8/8/8/8 w - - 0 1' --sims 100
python3 chess/cli.py perft 3
```

The last command counts 8,902 legal move paths from the initial position. FEN loading, legal UCI move parsing, and FEN export are exposed on `GameState` in `chess/scr/chess_engine.py`.

Implemented rules include king safety, castling, en passant, all four promotions, exact undo, checkmate, stalemate, repetition, the fifty-move counter, and common insufficient-material cases. Search copies preserve draw history. The application automatically accepts an available threefold or fifty-move draw; it does not implement a separate player draw-claim interface. Checkmate takes precedence. General dead-position proof is not implemented.

## Use with a UCI chess GUI

Select the executable `chess-engine` in a chess GUI on macOS/Linux, or configure the equivalent command:

```bash
python3 /absolute/path/to/games/chess/cli.py uci
```

The launcher works from any working directory and prefers the repository's `.venv` when present. Supported commands include `uci`, `isready`, `ucinewgame`, `position startpos`, `position fen`, move histories, `go nodes`, `go movetime`, clock/increment budgets, `go infinite`, `stop`, and `quit`. `Simulations` and `CPuct` are configurable options. Here `nodes` counts MCTS simulations. Search remains interruptible and answers `isready` while thinking.

This first adapter does not implement `depth`, `mate`, `searchmoves`, or pondering. Use simulation or time limits. Timing is cooperative: current move generation or a neural batch finishes before cancellation. UCI currently uses the built-in evaluator; neural checkpoints work through the GUI or model-play script.

## Neural self-play

Python 3.12 is the tested neural environment. Install the optional dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/neural.txt
```

Run a small end-to-end smoke test (this creates a checkpoint, not a strong player):

```bash
.venv/bin/python chess/scr/main.py \
  --iterations 1 --sims 2 --max-moves 2 --hidden 8 --device cpu \
  --batch-size 2 --min-buffer 1 --updates 1 \
  --checkpoint-dir checkpoints/smoke
```

A longer experiment can start with:

```bash
.venv/bin/python chess/scr/main.py --iterations 100 --sims 100 --max-moves 200
```

Use `--help` for replay capacity, seeds, checkpoint intervals, threads, and device options. Each experiment gets its own `runs/<timestamp>-<id>/` folder, with checkpoints inside it by default. `--resume PATH` loads network weights; optimizer and replay state start fresh. CPU, CUDA, and Apple MPS are supported when available; automated neural checks use CPU.

Play a new checkpoint with either interface:

```bash
.venv/bin/python chess/UI_with_AI.py --model checkpoints/smoke/policy_value_net_1.pt
.venv/bin/python chess/scr/play_with_model.py --model checkpoints/smoke/policy_value_net_1.pt --you w --temp 0
```

**Checkpoint format changed:** the network now receives 20 input planes and produces 4,480 policy logits, including separate underpromotion actions. Older 13-plane/4,096-action checkpoints are rejected with an explanation and need retraining. Genuine draws and move-limit positions train value zero by default, while retaining their policy targets. `--cutoff-value-policy exclude` remains available to reproduce the value-mask experiment.

## Inspect how training learns

Run a short experiment with every game traced and repeated fixed-position checks:

```bash
.venv/bin/python chess/scr/main.py \
  --iterations 12 --sims 64 --max-moves 200 --hidden 128 \
  --batch-size 64 --min-buffer 256 --updates 2 --seed 1 --device cpu \
  --run-dir runs/inspection-01 --trace-every 1 \
  --cutoff-value-policy draw \
  --probe-every 4 --probe-sims 80 --save-every 4 --report-every 1
```

Open `runs/inspection-01/report.html` in a browser. The report is a local, self-contained file with no network dependencies. It separates real chess endings from move-limit cutoffs, plots optimizer diagnostics, compares raw network predictions with MCTS on fixed positions, and lets you inspect individual game decisions on a board. A run folder must be empty or new, so an earlier experiment cannot be overwritten accidentally.

Every game records a termination reason and UCI move history. Detailed position traces record the first game and every tenth game by default; `--trace-every 1` records all games and `0` disables detailed tracing. Fixed probes run at start, end, and every fifth game by default; `--probe-every 0` disables them. Reports refresh every five games and at completion or failure. Logs are written after each game/update and remain readable if a run is interrupted. To rebuild a report without training:

```bash
python3 chess/scr/report.py runs/inspection-01
```

The files are designed for direct inspection or later analysis:

| File | Contents |
|---|---|
| `config.json`, `source/` | Hyperparameters, seed, software versions, source hashes and a copy of the exact training source |
| `games.jsonl` | Actual ending, outcome, final FEN, moves, length, time, and optional per-move predictions/search traces |
| `updates.jsonl` | Every optimizer update: losses, policy KL/agreement, eligible value predictions/targets, gradient norm, and separate policy/value sample counts |
| `probes.jsonl` | The same six positions before training and at later intervals; separate network and search behavior |
| `summary.json` | Current run status, ending counts, cutoff fractions, replay composition, and checkpoints |
| `report.html` | Interactive view of those records |

Cutoffs use chess outcome `*` (unfinished). By default their positions train both policy and value, with zero-valued labels for value training (`--cutoff-value-policy draw`). The experimental `--cutoff-value-policy exclude` retains their policy targets but omits them from value loss, averaging value MSE over the remaining eligible positions. The report shows the active cutoff policy and the number of examples supervising each head. Historical runs without a cutoff-policy flag also use draw labels. Metrics are observations of the current learning setup; fixed tactical checks and lower loss do not establish an Elo rating. See [the inspection guide](docs/training-inspection.md) for how to interpret the metrics.

The [first recorded experiment](experiments/2026-11-06-inspection-01/README.md) includes configuration, game histories, update/probe measurements, and a learning figure: 8 checkmates and 4 cutoffs, with no demonstrated value-prediction advantage over always predicting zero.

The [matched cutoff-value comparison](experiments/2026-11-06-cutoff-value-mask-01/README.md) reruns that setup while excluding unfinished positions from value loss. All 800 cutoff positions remain available for policy training. Both models solve 2/4 tactical checks and neither beats always-zero value predictions on the six common labeled examples; the small run does not establish a learning benefit. The reusable `scr.compare_training` command checks matching initialization/configuration and compares saved models on identical inputs.

## Tests

The core and public interfaces can be checked without dependencies:

```bash
python3 -m unittest discover -s tests -v
```

For independent rules comparison and neural tests:

```bash
.venv/bin/python -m pip install -r requirements/dev.txt -r requirements/neural.txt
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover six reference perft positions, special moves, undo/history, tactical MCTS, value perspective, batched neural evaluation, temperature, checkpoints, and subprocess UCI behavior. The optional [python-chess](https://python-chess.readthedocs.io/en/latest/) oracle compares legal moves/state across seeded games and drives the UCI executable. Runtime rules and search remain custom code. GitHub Actions runs core checks on Python 3.12/3.14 and neural checks on 3.12.

The `requirements/*py312.lock.txt` and `*py315.lock.txt` files are historical environment snapshots; use `gui.txt`, `neural.txt`, and `dev.txt` for the current setup. Virtual environments and model weights are ignored by Git.

## Development direction

- Establish search strength with repeatable tactical suites and engine matches.
- Improve evaluation and search efficiency, then benchmark neural search against the baseline.
- Add checkpoint evaluation and optimizer/replay persistence before scaling training.
- Extend UCI coverage and add PGN import/export.

See [MCTS implementation notes](docs/mcts.md) for value conventions and search stages. UCI communication follows the supported subset of the [UCI specification](https://backscattering.de/chess/uci/).
