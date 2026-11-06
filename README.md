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

Use `--help` for replay capacity, seeds, checkpoint intervals, threads, and device options. Checkpoints go to `chess/scr/checkpoints/` by default. `--resume PATH` loads network weights; optimizer and replay state start fresh. CPU, CUDA, and Apple MPS are supported when available; automated neural checks use CPU.

Play a new checkpoint with either interface:

```bash
.venv/bin/python chess/UI_with_AI.py --model checkpoints/smoke/policy_value_net_1.pt
.venv/bin/python chess/scr/play_with_model.py --model checkpoints/smoke/policy_value_net_1.pt --you w --temp 0
```

**Checkpoint format changed:** the network now receives 20 input planes and produces 4,480 policy logits, including separate underpromotion actions. Older 13-plane/4,096-action checkpoints are rejected with an explanation and need retraining. Training assigns zero to draws and truncated games; stronger training will need longer games, evaluation matches, and careful treatment of truncation.

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
