""" Generate a graphics test of tkinter for the module chess_engine."""
import tkinter as tk                # for the GUI
from chess_engine import GameState  # our chess engine module
from PIL import Image, ImageTk      # for image manipulation and use in tkinter
import os                           # for accessing image files found in a different folder

WINDOW_NAME = "Chess"           # set window name
SQUARE_LENGTH = 60              # set square length
game = GameState()              # create instance of GameState class
root = tk.Tk()                  # create instance of the Tk class
root.title(WINDOW_NAME)         # sets title of the window
board_frm = tk.Frame(root)      # create frame for the chess board
info_frm = tk.Frame(root)       # create frame for text information
two_click_history_L = []          # init history for user left clicks
two_click_history_R = []

# position board and info in frame
board_frm.grid(row = 0, column=0)
info_frm.grid(row=0, column=1)

# add textbox into info_frame
text_box = tk.Label(info_frm, text = "White goes first.", font= ("Courier", 12))
text_box.pack()

# load in images 
image_folder_path = "chess/scr/chess_piece_images"
piece_names = ["bB", "bK", "bN", "bP", "bQ", "bR", "wB", "wK", "wN", "wP", "wQ", "wR"]

# create dictionary for accessing image path names by piece_name
image_path_names  = {} # dictionary of {"bB": "chess_piece_images/bB.png", ...}
for piece_name in piece_names:
    image_path_names[piece_name] = os.path.join(image_folder_path , piece_name + ".png")

# create dictonary for accessing images from piece name
piece_photos = {} # dictionary of {"piece_name": photo, ...}
for piece_name in piece_names:
    image = Image.open(image_path_names[piece_name]).resize((SQUARE_LENGTH, SQUARE_LENGTH))
    photo = ImageTk.PhotoImage(image)
    piece_photos[piece_name] = photo


# create a function to draw the underlying board
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

"""We need to initialze the board here! The functions after this depend on our canvas_grid"""        
canvas_grid = draw_board(board_frm)         # draw initial board

# add pieces to the board
def draw_pieces(board):
    """This function draws the pieces on the board on the same frame of a background canvas_grid
    board = GameState().board
    canvas_grid = draw_board(frame)"""
    for row in range(len(board)):
        for col in range(len(board)):
            if board[row][col] != "--": # if the square is not empty
                piece_name = board[row][col]
                canvas_grid[row][col].create_image(int(SQUARE_LENGTH/2), 
                                                   int(SQUARE_LENGTH/2), 
                                                   image = piece_photos[piece_name])
                canvas_grid[row][col].image = piece_photos[piece_name]
            else:
                pass

draw_pieces(board=game.board)               # draw pieces on the board

# clearing canvas_grid utility
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

# update textual game info on side
def update_text_info(game):
    """Update the text information about the game in text_box label"""
    row_str_len = len(str(["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"]))
    info_text = "Current player is: " + game.player
    info_text = info_text + "\nTurn number is: " + str(game.turn) +"\n\n\n"

    # add an asthetic divider
    for _ in range(row_str_len):
        info_text = info_text + "="
    info_text = info_text + "\n"

    # display current board string
    info_text = info_text + game.get_board_string()

    # add an asthetic divider
    for _ in range(row_str_len):
        info_text = info_text + "="
    info_text = info_text + "\n"

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

def highlight_background(board_coord):
    """highlight the background on canvas_grid with a specified color"""
    row_index, col_index = board_coord
    canvas_grid[row_index][col_index].config(bg = "#8FC500")

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
        
# add event handler for left clicks
def on_left_click(event):
    """This function tells the GUI what to do with clicks
    It includes some logic to handle clicking the board to move pieces
    it uses is_valid_move from the module chess_engine, clear_pieces, and draw_pieces"""
    
    two_click_history_L.append(find_coords(event)) 

    # if first click then highlight_border canvas
    if len(two_click_history_L) == 1:
        highlight_border(two_click_history_L[0])

    # check if two clicks have been registered
    if len(two_click_history_L) == 2:                
        if game.is_valid_move(two_click_history_L):   # check if valid chess move
            game.move_piece(two_click_history_L)      # move_pieces accordingly
            clear_photos(two_click_history_L)        # clear the images from canvas_grid
            draw_pieces(board=game.board)           # draw new positions on canvas_grid
            update_text_info(game)                  # update text info about the game
        reset_border(two_click_history_L[0])  # reset border even if move isn't valid
        reset_background()
        two_click_history_L.clear()           # clear click history 

# add event handler for right clicks
def on_right_click(event):
    coords = find_coords(event)
    valid_moves = game.get_valid_moves(coords)

    two_click_history_R.append(coords)

    # if one click is registered, highlight avaiable moves
    if len(two_click_history_R) == 1:
        for move in valid_moves:
            highlight_background(move)

    if len(two_click_history_R) == 2:
        reset_background()
        two_click_history_R.clear()

    

# bind mouse clics to functions
for canvas_row in canvas_grid:
    for square in canvas_row:
        square.bind("<Button-1>", on_left_click)
        square.bind("<Button-3>", on_right_click)

# run main event loop for Tk
root.mainloop()