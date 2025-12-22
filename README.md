# Games

This project contains basic game engines written in Python with minimal package dependencies. Currently only chess is completed. 

## Getting Started

First, clone the github repository by running the following code in the terminal:

`git clone https://github.com/Hydrodynamical/games.git`

Then navigate to the game you would like to try out. For example 

`cd chess`

and then to run the program 

`python main.py`


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

- [ ] **Correct undo functionality**
  - Store captured piece in move log
  - Restore board state exactly on undo
  - Prepare undo logic for future MCTS/search (no deepcopies)

---

### Game State & Feedback
- [ ] **Add “White/Black is in check” display**
  - Use attack-map logic (not legal moves)
  - Update after every move

- [ ] **Highlight capture moves**
  - Display available capture moves in a distinct color (e.g. red)
  - Non-capture moves remain current highlight color
  - Use `is_attack=False` move generation + board inspection

---

### UI / Interaction
- [ ] **Add undo button**
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

