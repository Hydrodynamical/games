# Games

This project contains basic game engines written in Python with minimal package dependencies. Currently only chess is completed. 

## Getting Started

First, clone the github repository by running the following code in the terminal:

`git clone https://github.com/Hydrodynamical/games.git`

Then navigate to the game you would like to try out. For example 

`cd chess`

and then to run the program 

`python main.py`

## Reproducible Python Environments

This repo intentionally does **not** commit virtual environment folders (they are ignored via `.gitignore`).
Instead, it tracks **lock files** generated from each virtual environment so you can recreate them deterministically.

Currently there are two environments:

- **Python 3.12** (For ML): `requirements/requirements-py312.lock.txt`
- **Python 3.15 (alpha)** (For UI): `requirements/requirements-py315.lock.txt`

### Create + Install (Python 3.12 for ML)

```bash
cd /Users/jkmiller/games
python3.12 -m venv .venv312
./.venv312/bin/python -m pip install -U pip
./.venv312/bin/python -m pip install -r requirements/requirements-py312.lock.txt
```

### Create + Install (Python 3.15 alpha / experimental for UI)

Only use this if you already have a Python 3.15 alpha installed locally.

```bash
cd /Users/jkmiller/games
python3.15 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r requirements/requirements-py315.lock.txt
```

### Updating the Lock Files

When you change dependencies inside an environment, regenerate its lock file:

```bash
cd /Users/jkmiller/games
./.venv312/bin/python -m pip freeze --exclude-editable | sort > requirements/requirements-py312.lock.txt
./.venv/bin/python -m pip freeze --exclude-editable | sort > requirements/requirements-py315.lock.txt
```

Commit the updated files under `requirements/` and push to GitHub.


## Next Iteration TODO

### Core Rules & Engine Correctness
- [ ] **Add castling**
  - Track castling rights (K/Q side for both colors)
  - Disallow castling through check or out of check
  - Update legality filtering to account for attacked transit squares

- [ ] **Add en passant**
  - Track en passant target square
  - Allow capture only on the immediately following move
  - Ensure en passant does not expose own king to check

- [ ] **Add promotion**
  - Default to queen promotion
  - UI: Popup with prompt for string input "Q,N,B,R,".

- [!] **Correct undo functionality**
  - Store captured piece in move log
  - Restore board state exactly on undo
  - Prepare undo logic for future MCTS/search (no deepcopies)

- [ ] **Finite game checks (beyond stalemate)**
  - 50 move rule
  - Threefold repetition
  - Insufficient material
---

### Game State & Feedback
- [ ] **Add “White/Black is in check” display**
  - Use attack-map logic (not legal moves)
  - Update after every move

- [!] **Highlight capture moves**
  - Display available capture moves in a distinct color (e.g. red)
  - Non-capture moves remain current highlight color
  - Use `is_attack=False` move generation + board inspection

---

### UI / Interaction
- [!] **Add undo button**
  - Revert last move using corrected undo logic
  - Sync board, player, and move history

- [ ] **Add move navigation**
  - Arrow keys (← / →) to step through move history
  - Support replaying from any previous position

---

### Output / Export
- [ ] **Save game history as a GIF**
  - Render board after each move
  - Allow custom filename input
  - Optional: configurable frame delay

