"""Play against MCTS in Tk; optionally provide a trained neural checkpoint."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from queue import Empty, Queue
import random
from threading import Thread

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk

from scr.mcts import mcts_search, best_move
from scr.chess_engine import GameState

parser = argparse.ArgumentParser(description=__doc__)
opponent_args = parser.add_mutually_exclusive_group()
opponent_args.add_argument("--model", help="Optional version-2 neural checkpoint")
opponent_args.add_argument("--opponent", choices=("supervised", "selfplay"), help="Choose a trained opponent preset")
parser.add_argument("--you", choices=("w", "b"), default="w")
parser.add_argument("--sims", type=int, default=200)
parser.add_argument("--seconds", type=float, default=2.0, help="Maximum search seconds per move; 0 uses only the simulation limit")
parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
parser.add_argument("--threads", type=int, default=4, help="CPU threads for neural inference")
args = parser.parse_args()
if args.sims < 0 or not math.isfinite(args.seconds) or args.seconds < 0:
    parser.error("--sims and --seconds must be nonnegative and finite")
if args.threads < 1:
    parser.error("--threads must be positive")

repo_root = Path(__file__).resolve().parents[1]
opponents = {
    "handcrafted": {"label": "Built-in MCTS", "path": None},
    "supervised": {"label": "Hikaru supervised (epoch 3)",
                   "path": repo_root / "runs/2026-11-06-hikaru-supervised-02/checkpoints/best.pt"},
    "selfplay": {"label": "Self-play (100 games)",
                 "path": repo_root / "runs/2026-11-06-hikaru-selfplay-01/checkpoints/policy_value_net_100.pt"},
}
if args.model:
    opponents["custom"] = {"label": "Custom checkpoint", "path": Path(args.model).expanduser().resolve()}
current_opponent = args.opponent or ("custom" if args.model else "handcrafted")
human_color = args.you


def load_opponent(key):
    path = opponents[key]["path"]
    if path is None:
        return None
    import torch
    from scr.rl_interface import load_model
    torch.set_num_threads(args.threads)
    return load_model(path, device=None if args.device == "auto" else args.device)


WINDOW_NAME = "Chess — MCTS"
SQUARE_LENGTH = 60
game = GameState()
AI_ENABLED = True
AI_PLAYS = "b" if args.you == "w" else "w"
AI_USE_MCTS = True
AI_SIMS = args.sims
AI_TEMP = 0.0
AI_THINKING = False
ai_results = Queue()
try:
    model = load_opponent(current_opponent)
except Exception as exc:
    parser.error(f"Could not load {opponents[current_opponent]['label']}: {exc}")
print(f"Opponent: {opponents[current_opponent]['label']}")

root = tk.Tk()                  # create instance of the Tk class
root.title(WINDOW_NAME)         # sets title of the window
board_frm = tk.Frame(root)      # create frame for the chess board
info_frm = tk.Frame(root)       # create frame for text information
two_click_history_L = []        # init history for user left clicks
two_click_history_R = []        # init history for user right clicks
two_click_history_M = []        # init history for user middle clicks

# print Tk version
print("Tk version =", root.tk.call("info", "patchlevel"))

# position board and info in frame
board_frm.grid(row = 0, column=0)
info_frm.grid(row=0, column=1, sticky="n", padx=12, pady=8)

# Choices describe the next game; the active opponent remains visible below.
settings_frm = tk.Frame(info_frm)
settings_frm.pack(fill="x", pady=(0, 8))
tk.Label(settings_frm, text="Opponent for next game:").grid(row=0, column=0, columnspan=2, sticky="w")
opponent_choice = tk.StringVar(value=opponents[current_opponent]["label"])
opponent_picker = ttk.Combobox(settings_frm, textvariable=opponent_choice, state="readonly", width=34,
                              values=[entry["label"] for entry in opponents.values()])
opponent_picker.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
tk.Label(settings_frm, text="You play:").grid(row=2, column=0, sticky="w")
color_choice = tk.StringVar(value="White" if human_color == "w" else "Black")
color_picker = ttk.Combobox(settings_frm, textvariable=color_choice, state="readonly", width=12,
                           values=("White", "Black"))
color_picker.grid(row=2, column=1, sticky="e")
new_game_button = tk.Button(settings_frm, text="New game", command=lambda: new_game())
new_game_button.grid(row=3, column=0, sticky="ew", pady=8)
undo_button = tk.Button(settings_frm, text="Undo", command=lambda: undo_move())
undo_button.grid(row=3, column=1, sticky="ew", pady=8)
opponent_status = tk.Label(info_frm, justify="left", anchor="w", font=("Courier", 11))
opponent_status.pack(fill="x")
tk.Label(info_frm, text="Choices apply when you press New game.\n\nLeft click to move; right click for legal moves.\n"
         "u: undo | a: AI move | t: toggle AI | m: toggle MCTS",
         justify="left", font=("Courier", 10)).pack(pady=(8, 8))

# add textbox into info_frame
text_box = tk.Label(info_frm, text="White to move.", font=("Courier", 12))
text_box.pack()

# ----------------------------
# NEW: AI info panel (top moves, value, etc.)
# ----------------------------
ai_box = tk.Text(info_frm, height=18, width=44, font=("Courier", 11))
ai_box.pack(pady=(10, 0))
ai_box.insert("1.0", "AI analysis appears after its move.\n")
ai_box.config(state="disabled")

def set_ai_info(text: str) -> None:
    """Replace contents of the AI info panel."""
    ai_box.config(state="normal")
    ai_box.delete("1.0", "end")
    ai_box.insert("1.0", text)
    ai_box.config(state="disabled")


def update_opponent_status():
    color_name = "White" if human_color == "w" else "Black"
    limit = f"{args.seconds:g}s cap" if args.seconds else "no time cap"
    search_detail = (f"MCTS: up to {AI_SIMS} simulations, {limit}"
                     if AI_USE_MCTS or model is None else "Raw neural policy (MCTS off)")
    runtime = f"{next(model.parameters()).device}, {args.threads} CPU threads" if model is not None else "built-in evaluator"
    opponent_status.config(text=f"Opponent: {opponents[current_opponent]['label']}\nYou: {color_name}\n"
                           f"{search_detail}\n{runtime}\nAI: {'enabled' if AI_ENABLED else 'disabled'}")
    root.title(f"Chess — {opponents[current_opponent]['label']} — You play {color_name}")


def set_game_controls_enabled(enabled):
    for picker in (opponent_picker, color_picker):
        picker.configure(state="readonly" if enabled else "disabled")
    for button in (new_game_button, undo_button):
        button.configure(state="normal" if enabled else "disabled")


update_opponent_status()

# load in images using a path relative to this file's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(script_dir, "scr", "chess_piece_images")
piece_names = ["bB", "bK", "bN", "bP", "bQ", "bR", "wB", "wK", "wN", "wP", "wQ", "wR"]

# create dictionary for accessing image path names by piece_name
image_path_names  = {} # dictionary of {"bB": "chess_piece_images/bB.png", ...}
for piece_name in piece_names:
    image_path_names[piece_name] = os.path.join(image_folder_path , piece_name + ".png")

# create dictionary for accessing images from piece name
piece_photos = {} # dictionary of {"piece_name": photo, ...}
for piece_name in piece_names:
    image = Image.open(image_path_names[piece_name]).resize((SQUARE_LENGTH, SQUARE_LENGTH))
    photo = ImageTk.PhotoImage(image)
    piece_photos[piece_name] = photo

def draw_board(frame, 
               board_length = 8, 
               light_color = "#ffce9e", 
               dark_color = "#d88c44",
               square_length = SQUARE_LENGTH,
               border_thickness = 2,
               border_color = "black"):
    """ Create a function which adds the board to the specified frame. 
        Returns: canvas_grid[row][col]  """
    canvas_grid = [[None for _ in range(board_length)] for _ in range(board_length)] 
    for row in range(board_length):
         for col in range(board_length):
            if (row + col)%2 == 0: # white squares, upper left is white
                square = tk.Canvas(frame, 
                                        width=square_length, 
                                        height=square_length,
                                        background=light_color,
                                        highlightbackground=border_color,
                                        highlightthickness= border_thickness)
                square.grid(row=row, column=col)
                canvas_grid[row][col] = square
            else:
                square = tk.Canvas(frame, 
                                        width=square_length, 
                                        height=square_length,
                                        background=dark_color,
                                        highlightbackground=border_color,
                                        highlightthickness= border_thickness)
                square.grid(row=row, column=col)
                canvas_grid[row][col] = square
    return canvas_grid

"""
We need to draw the board now! The functions after this depend on the canvas_grid.
"""        

canvas_grid = draw_board(board_frm)         # draw initial board


def orient_board():
    # canvas_grid stays indexed by board coordinates; only visual placement changes.
    for row, canvases in enumerate(canvas_grid):
        for col, canvas in enumerate(canvases):
            canvas.grid(row=row if human_color == "w" else 7 - row,
                        column=col if human_color == "w" else 7 - col)


orient_board()

def draw_pieces(board):
    """This function draws the pieces on the board on the same frame of a background canvas_grid
    board = GameState().board
    canvas_grid = draw_board(frame)"""
    # Remove all existing piece sprites from the canvas
    for row in range(len(board)):
        for col in range(len(board)):
            # clear existing images first
            canvas_grid[row][col].delete("all") # needed to implement undo move
            if board[row][col] != "--": # if the square is not empty
                piece_name = board[row][col]
                canvas_grid[row][col].create_image(int(SQUARE_LENGTH/2), 
                                                   int(SQUARE_LENGTH/2), 
                                                   image = piece_photos[piece_name])
                canvas_grid[row][col].image = piece_photos[piece_name]
            else:
                pass

draw_pieces(board=game.board)               # draw pieces on the board

def clear_photos(positions):
    """Clear the image at positions= [[row, col], ...] on the canvas_grid"""
    for position in positions:
        row, col = position
        photo_ids = canvas_grid[row][col].find_all()
        for photo_id in photo_ids:
            canvas_grid[row][col].delete(photo_id)

def find_coords(event):
    """for finding row index and column index after a click
    Returns [row_index, col_index]"""
    clicked_canvas = event.widget # name widget that was clicked on
    for row_index, row in enumerate(canvas_grid):
        if clicked_canvas in row:
            col_index = row.index(clicked_canvas)
            break #break out of loop if clicked canvas is not in row
    return [row_index, col_index]

def update_text_info(game, show_board_string = False, checkmate = False, stalemate = False, extra_text: str = ""):
    """Update the text information about the game in text_box label"""
    row_str_len = len(str(["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"]))
    info_text = "Current player is: " + game.player
    info_text = info_text + "\nTurn number is: " + str(game.turn) +"\n\n\n"

    # display current board string
    if show_board_string:
        # add an aesthetic divider
        for _ in range(row_str_len):
            info_text = info_text + "="
        info_text = info_text + "\n"
        info_text = info_text + game.get_board_string()
        # add an aesthetic divider
        for _ in range(row_str_len):
            info_text = info_text + "="
        info_text = info_text + "\n"

    # if checkmate occurs print message
    if checkmate:
        info_text = info_text + "\nCheckmate!\n\n"
        info_text = info_text + "Player " + str(game.opponent_color()) + " has won."

    elif stalemate:
        info_text = info_text + "\nStalemate!\n\n"
    elif game.is_draw():
        info_text += f"\nDraw: {game.draw_reason()}\n"

    if extra_text:
        info_text += "\n" + extra_text

    # add the resulting info_text to the text_box
    text_box.config(text = info_text, font = ("Courier", 12))
     
def highlight_border(board_coord):
    """Highlight canvas specified by board_coord = [row_index, col_index]
    Uses canvas_grid"""
    row_index, col_index = board_coord
    canvas_grid[row_index][col_index].config(highlightbackground='red')

def reset_border(board_coord):
    """Highlight canvas specified by board_coord = [row_index, col_index]
    Uses canvas_grid"""
    row_index, col_index = board_coord
    canvas_grid[row_index][col_index].config(highlightbackground= "black")

def highlight_background(board_coord, color = "#4CC500"):
    """highlight the background on canvas_grid with a specified color"""
    row_index, col_index = board_coord
    canvas_grid[row_index][col_index].config(bg = color)

def reset_background(board_dimension = 8):
    """reset the background on canvas_grid to it's original color.
    light_color = "#ffce9e", 
    dark_color = "#d88c44"."""
    for row in range(board_dimension):
        for col in range(board_dimension):
            if (row + col) % 2 == 0:
                canvas_grid[row][col].config(bg = "#ffce9e")
            else:
                canvas_grid[row][col].config(bg = "#d88c44")

def _move_to_str(mv) -> str:
    if hasattr(mv, "uci"):
        return mv.uci()
    if hasattr(mv, "start") and hasattr(mv, "end"):
        pm = getattr(mv, "piece_moved", "")
        pc = getattr(mv, "piece_captured", "")
        cap = f" x{pc}" if pc not in ("--", "", None) else ""
        return f"{pm}:{tuple(mv.start)}->{tuple(mv.end)}{cap}"
    return str(mv)


def _pick_from_move_probs(pi_moves: dict, temperature: float = 0.0):
    if not pi_moves:
        return None
    if temperature == 0:
        return max(pi_moves, key=pi_moves.get)
    maximum = max(pi_moves.values())
    weights = [math.exp((math.log(p) - math.log(maximum)) / temperature) if p else 0.0
               for p in pi_moves.values()]
    return random.choices(list(pi_moves), weights=weights, k=1)[0]


def on_left_click(event):
    """This function tells the GUI what to do with left clicks
    It includes the logic to handle clicking the board to move pieces"""
    
    if AI_THINKING or game.is_checkmate() or game.is_stalemate() or game.is_draw():
        return
    if AI_ENABLED and game.player == AI_PLAYS:
        return
    # keep track of click coordinates
    two_click_history_L.append(find_coords(event)) 

    # if first click then highlight_border canvas
    if len(two_click_history_L) == 1:
        highlight_border(two_click_history_L[0])

    # check if two clicks have been registered
    if len(two_click_history_L) == 2:
        if game.is_legal_move(two_click_history_L):             # check if valid chess move
            start, end = map(tuple, two_click_history_L)
            candidates = [mv for mv in game.get_all_legal_moves(game.player, as_moves=True)
                          if mv.start == start and mv.end == end]
            mv = candidates[0]
            if len(candidates) > 1:
                choice = simpledialog.askstring("Promotion", "Promote to Q, R, B or N:", initialvalue="Q", parent=root)
                if choice is None:
                    reset_border(two_click_history_L[0])
                    two_click_history_L.clear()
                    return
                promotion = choice.strip().upper()
                selected = next((move for move in candidates if move.promotion == promotion), None)
                if selected is None:
                    reset_border(two_click_history_L[0])
                    two_click_history_L.clear()
                    return
                mv = selected
            game.make_move(mv)                                  # make the move in the game state
            clear_photos(two_click_history_L)                   # clear the images from canvas_grid
            draw_pieces(board=game.board)                       # draw new positions on canvas_grid
            update_text_info(game, checkmate=game.is_checkmate(), stalemate=game.is_stalemate())
            root.after(100, ai_make_move)
        reset_border(two_click_history_L[0])                    # reset border even if move isn't valid
        reset_background()                                      # reset the background of all squares
        two_click_history_L.clear()                             # clear click history after two clicks

# NEW version of on_right_click
def on_right_click(event):
    """Right click: highlight legal moves. Captures in red, quiet moves in green."""
    coords = find_coords(event)

    two_click_history_R.append(coords)

    if len(two_click_history_R) == 1:
        # Use Move objects so we can color captures differently
        legal_moves = game.get_legal_moves(coords)  # returns List[Move]

        for mv in legal_moves:
            end_sq = list(mv.end)
            if mv.piece_captured != "--":
                highlight_background(end_sq, color="#C54800") # for captures
            else:
                highlight_background(end_sq)  # default green

    if len(two_click_history_R) == 2:
        reset_background()
        two_click_history_R.clear()

# NEW: bind 'u' and 'Backspace' to undo move
def new_game():
    global game, model, current_opponent, human_color, AI_PLAYS, AI_ENABLED
    if AI_THINKING:
        return
    selected = next(key for key, entry in opponents.items() if entry["label"] == opponent_choice.get())
    try:
        selected_model = load_opponent(selected)
    except Exception as exc:
        messagebox.showerror("Cannot load opponent", f"{opponents[selected]['label']} could not be loaded.\n\n{exc}", parent=root)
        return
    for history in (two_click_history_L, two_click_history_R, two_click_history_M):
        for square in history:
            reset_border(square)
        history.clear()
    model, current_opponent = selected_model, selected
    human_color = "w" if color_choice.get() == "White" else "b"
    AI_PLAYS = "b" if human_color == "w" else "w"
    AI_ENABLED = True
    game = GameState()
    orient_board()
    reset_background()
    draw_pieces(game.board)
    update_opponent_status()
    update_text_info(game)
    set_ai_info("New game ready.\nThe AI plays the opposite color.\n")
    root.after(100, ai_make_move)


def undo_move(event=None):
    if AI_THINKING:
        return
    previous_plies = len(game.move_log)
    for history in (two_click_history_L, two_click_history_R, two_click_history_M):
        for square in history:
            reset_border(square)
        history.clear()
    game.undo_last_move()
    if AI_ENABLED and game.player == AI_PLAYS:
        game.undo_last_move()
    reset_background()
    draw_pieces(board=game.board)
    update_text_info(game, checkmate=game.is_checkmate(), stalemate=game.is_stalemate())
    set_ai_info(("Position restored by Undo." if previous_plies else "No moves to undo.")
                + "\nAI analysis cleared.\n")
    if AI_ENABLED and game.player == AI_PLAYS:
        root.after(100, ai_make_move)

root.bind("<u>", undo_move)             # press 'u' to undo
root.bind("<BackSpace>", undo_move)  # press 'Backspace' to undo

def on_press_p(event=None):
    game.print_board()
    game.print_move_log(last_n_moves=30)

root.bind("<p>", on_press_p)

# bind mouse clicks to functions
for canvas_row in canvas_grid:
    for square in canvas_row:
        square.bind("<Button-1>", on_left_click)
        square.bind("<Button-3>", on_right_click)


def ai_make_move(event=None):
    """Run search in a worker so the board remains responsive."""
    global AI_THINKING
    if AI_THINKING or not AI_ENABLED or AI_PLAYS not in ("both", game.player):
        return
    if game.is_checkmate() or game.is_stalemate() or game.is_draw():
        return
    reset_background()
    AI_THINKING = True
    set_game_controls_enabled(False)
    snapshot = game.fast_copy_for_search()
    search_model, use_mcts, search_sims = model, AI_USE_MCTS, AI_SIMS
    search_seconds = args.seconds or None
    set_ai_info(f"{opponents[current_opponent]['label']} is thinking…")

    def search():
        try:
            if search_model is not None:
                from scr.rl_interface import critic_value, policy_distribution
            value = float(critic_value(search_model, snapshot)) if search_model is not None else None
            if use_mcts or search_model is None:
                probabilities, search_root = mcts_search(
                    snapshot, search_model, num_sims=search_sims, time_limit=search_seconds,
                    show_progress=False, return_root=True,
                )
            else:
                probabilities = policy_distribution(search_model, snapshot)
                search_root = None
            ai_results.put((probabilities, value, search_root, None))
        except Exception as exc:
            ai_results.put(({}, None, None, str(exc)))

    Thread(target=search, daemon=True).start()
    root.after(30, finish_ai_move)


def finish_ai_move():
    global AI_THINKING
    try:
        pi_moves, value, search_root, error = ai_results.get_nowait()
    except Empty:
        root.after(30, finish_ai_move)
        return
    AI_THINKING = False
    set_game_controls_enabled(True)
    if error:
        set_ai_info(f"Search failed: {error}")
        return
    mv = best_move(pi_moves, search_root) if AI_TEMP == 0 else _pick_from_move_probs(pi_moves, AI_TEMP)
    if mv is None:
        return
    game.make_move(mv)
    draw_pieces(game.board)
    update_text_info(game, checkmate=game.is_checkmate(), stalemate=game.is_stalemate(),
                     extra_text=f"AI last move: {_move_to_str(mv)}")
    search_info = (f"MCTS simulations completed: {sum(edge.n for edge in search_root.edges.values())}"
                   if search_root is not None else "Raw neural policy (MCTS off)")
    lines = [search_info, f"Chosen move: {_move_to_str(mv)}"]
    if value is not None:
        lines.append(f"Neural value (AI perspective): {value:+.3f}")
    lines += ["", "Top candidate moves:"]
    top = sorted(pi_moves.items(), key=lambda item: item[1], reverse=True)[:10]
    lines += [f"{probability:7.4f}  {_move_to_str(move)}" for move, probability in top]
    set_ai_info("\n".join(lines))


def toggle_ai(event=None):
    global AI_ENABLED
    if AI_THINKING:
        return
    AI_ENABLED = not AI_ENABLED
    update_opponent_status()
    print("AI_ENABLED =", AI_ENABLED)
    if AI_ENABLED:
        root.after(100, ai_make_move)

def toggle_mcts(event=None):
    global AI_USE_MCTS
    if AI_THINKING:
        return
    AI_USE_MCTS = not AI_USE_MCTS
    update_opponent_status()
    print("AI_USE_MCTS =", AI_USE_MCTS)

root.bind("<a>", ai_make_move)   # press 'a' to make the AI play one move now
root.bind("<t>", toggle_ai)      # toggle AI enabled
root.bind("<m>", toggle_mcts)    # toggle MCTS on/off

# run main event loop for Tk
root.after(100, ai_make_move)
root.mainloop()
