""" Generate a graphics test of tkinter for the module chess_engine."""
import tkinter as tk                    # for the GUI
from scr.chess_engine import GameState  # our chess engine module
from PIL import Image, ImageTk          # for image manipulation and use in tkinter
import os                               # for accessing image files found in a different folder

WINDOW_NAME = "Chess"           # set window name
SQUARE_LENGTH = 60              # set square length
game = GameState()              # create instance of GameState class
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

def update_text_info(game, show_board_string = False, checkmate = False, stalemate = False, draw_reason = None):
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

     # Terminal messages (draw > checkmate > stalemate)
    if draw_reason:
        info_text += "\nDraw!\n\n"
        if draw_reason == "threefold_repetition":
            info_text += "Reason: Threefold repetition.\n"
        elif draw_reason == "fifty_move_rule":
            info_text += "Reason: 50-move rule.\n"
        else:
            info_text += f"Reason: {draw_reason}\n"

    # if checkmate occurs print message
    if checkmate:
        info_text = info_text + "\nCheckmate!\n\n"
        info_text = info_text + "Player " + str(game.opponent_color()) + " has won."

    elif stalemate:
        info_text = info_text + "\nStalemate!\n\n"

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

def get_draw_reason(game):
    """Return draw reason string or None."""
    if hasattr(game, "draw_reason"):
        return game.draw_reason()
    if hasattr(game, "is_draw") and game.is_draw():
        # fallback if you only implemented is_draw() without draw_reason()
        return "draw"
    return None


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
            draw_reason = get_draw_reason(game)
            update_text_info(game,                              # update text info about the game
                checkmate=(not draw_reason and game.is_checkmate()),
                stalemate=(not draw_reason and game.is_stalemate()),
                draw_reason=draw_reason,
            )
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
    draw_reason = get_draw_reason(game)
    update_text_info(game,
        checkmate=(not draw_reason and game.is_checkmate()),
        stalemate=(not draw_reason and game.is_stalemate()),
        draw_reason=draw_reason,
    )


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

# run main event loop for Tk
root.mainloop()
