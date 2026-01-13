""" Generate a graphics test of tkinter for the module chess_engine."""
import torch

from scr.mcts import mcts_search
from scr.rl_interface import PolicyValueNet, policy_distribution, critic_value

import tkinter as tk                    # for the GUI
from scr.chess_engine import GameState  # our chess engine module
from PIL import Image, ImageTk          # for image manipulation and use in tkinter
import os                               # for accessing image files found in a different folder

WINDOW_NAME = "Chess"           # set window name
SQUARE_LENGTH = 60              # set square length
game = GameState()              # create instance of GameState class
AI_ENABLED = True          # press 't' to toggle
AI_PLAYS = "both"             # set to "w" or "b" or "both"
AI_USE_MCTS = True         # press 'm' to toggle
AI_SIMS = 100              # MCTS sims per move
AI_TEMP = 0.0              # 0 = deterministic

# ----------------------------
# Model setup (loads once)
# ----------------------------

MODEL_PATH = "chess\scr\policy_value_net_3.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    _state = torch.load(MODEL_PATH, map_location=DEVICE)

    # If you saved a full model (rare here), _state will be an nn.Module.
    # If you saved a state_dict (your case), it will be an OrderedDict/dict.
    if hasattr(_state, "eval"):
        model = _state
    else:
        model = PolicyValueNet()
        model.load_state_dict(_state)

    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {MODEL_PATH} on {DEVICE}")
except Exception as e:
    model = None
    print(f"WARNING: Could not load model at {MODEL_PATH}: {e}")
# ----------------------------

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
info_frm.grid(row=0, column=1)

# add textbox into info_frame
text_box = tk.Label(info_frm, text = "White goes first.\n\n Left click to move.\n\nRight click to see available moves.", font= ("Courier", 12))
text_box.pack()

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
    if hasattr(mv, "start") and hasattr(mv, "end"):
        pm = getattr(mv, "piece_moved", "")
        pc = getattr(mv, "piece_captured", "")
        cap = f" x{pc}" if pc not in ("--", "", None) else ""
        return f"{pm}:{tuple(mv.start)}->{tuple(mv.end)}{cap}"
    return str(mv)


def _pick_from_move_probs(pi_moves: dict, temperature: float = 0.0):
    """Pick a move from {Move: prob}. temperature=0 -> argmax; else sample with p^(1/T)."""
    if not pi_moves:
        return None

    moves = list(pi_moves.keys())
    probs = torch.tensor([float(pi_moves[m]) for m in moves], dtype=torch.float32)

    if temperature == 0.0:
        return moves[int(torch.argmax(probs).item())]

    probs = torch.clamp(probs, min=1e-12)
    probs = probs ** (1.0 / float(temperature))
    probs = probs / probs.sum()
    idx = int(torch.multinomial(probs, 1).item())
    return moves[idx]


def on_left_click(event):
    """This function tells the GUI what to do with left clicks
    It includes the logic to handle clicking the board to move pieces"""
    
    # keep track of click coordinates
    two_click_history_L.append(find_coords(event)) 

    # if first click then highlight_border canvas
    if len(two_click_history_L) == 1:
        highlight_border(two_click_history_L[0])

    # check if two clicks have been registered
    if len(two_click_history_L) == 2:
        if game.is_legal_move(two_click_history_L):             # check if valid chess move
            mv = game.move_from_pair(two_click_history_L)       # create Move object from move_pair   
            game.make_move(mv)                                  # make the move in the game state
            clear_photos(two_click_history_L)                   # clear the images from canvas_grid
            draw_pieces(board=game.board)                       # draw new positions on canvas_grid
            update_text_info(game, checkmate=game.is_checkmate(), stalemate=game.is_stalemate())    # update text info about the game
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
def undo_move(event=None):
    game.undo_last_move()
    reset_background()
    draw_pieces(board=game.board)
    update_text_info(game, checkmate=game.is_checkmate(), stalemate=game.is_stalemate())

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
    """Have the model play one move for the current player (if enabled)."""
    global game

    if model is None:
        print("No model loaded.")
        return

    if not AI_ENABLED:
        return

    if AI_PLAYS not in ("both", game.player):
        return

    # terminal?
    if game.is_checkmate() or game.is_stalemate():
        return

    reset_background()

    # Choose move
    if AI_USE_MCTS:
        pi_moves = mcts_search(game, model, num_sims=AI_SIMS, show_progress=True)
    else:
        pi_moves = policy_distribution(model, game)

    mv = _pick_from_move_probs(pi_moves, temperature=AI_TEMP)
    if mv is None:
        return

    # Apply
    game.make_move(mv)
    draw_pieces(board=game.board)
    extra = f"\nAI last move: {_move_to_str(mv)}"
    update_text_info(game, checkmate=game.is_checkmate(), stalemate=game.is_stalemate(), extra_text=extra)

    # Optional: print what it did + value
    try:
        v = float(critic_value(model, game))
        top = sorted(pi_moves.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("\nAI played:", _move_to_str(mv))
        print("Value (side to move):", f"{v:+.3f}")
        print("Top moves:")
        for m, p in top:
            print(f"  {p:7.4f}  {_move_to_str(m)}")
    except Exception:
        pass

def toggle_ai(event=None):
    global AI_ENABLED
    AI_ENABLED = not AI_ENABLED
    print("AI_ENABLED =", AI_ENABLED)

def toggle_mcts(event=None):
    global AI_USE_MCTS
    AI_USE_MCTS = not AI_USE_MCTS
    print("AI_USE_MCTS =", AI_USE_MCTS)

root.bind("<a>", ai_make_move)   # press 'a' to make the AI play one move now
root.bind("<t>", toggle_ai)      # toggle AI enabled
root.bind("<m>", toggle_mcts)    # toggle MCTS on/off

# run main event loop for Tk
root.mainloop()
