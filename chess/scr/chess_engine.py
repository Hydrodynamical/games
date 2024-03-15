'''This module contains a class GameState 
    GameState is responsible for 
    storing current state of the game,
    determining valid move states, 
    storing game history,
    checking if move is valid.

'''



class GameState():

    # Initialize the class and store info about variables of game
    def __init__(self):
        # piece encoding:
        # "b" = black
        # "w" = white
        # "R" = rook
        # "N" = knight
        # "B" = bishop
        # "Q" = queen
        # "K" = king
        # "P" = pawn
        # "--" = empty_square
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]]
        
        # player is either white or black, and white starts first
        self.player = "w" 

        # turn number starts at 0
        self.turn = 0

        # board history is empty list
        self.history = []

    def print_board(self):
        '''Prints the board, the current player'''
        board_length = len(self.board)

        # print board
        for row in range(board_length):
            row_string = str(self.board[row])
            print(row_string + "\n")
        
        # print current player
        print("Current player is: " + self.player)

        # print turn number
        print("Turn number = " + str(self.turn))

        # print asthetic divider
        row_string_len = len(str(self.board[0]))
        divider = ""
        for _ in range(row_string_len):
            divider = divider + "="
        print(divider)
    
    def get_board_string(self):
        """Get the current board state as a string"""
        board_string = ""
        for row in self.board:
            board_string = board_string + str(row) + "\n"
        return board_string

    def switch_player(self):
        '''Switches player order'''
        if self.player == "w":
            self.player = "b"
        else:
            self.player = "w"

    def move_piece(self, move_pair):
        """This function moves a piece from move_pair[0] to move_pair[1]
        move_pair = [[row_0, col_0], [row_1, col_1]]
        """

        # get rows and columns of moves 
        row_0, col_0 = move_pair[0]
        row_1, col_1 = move_pair[1]

        # change board and append to history
        selected_piece = self.board[row_0][col_0]       # get the string of the selected piece
        self.board[row_0][col_0] = "--"                 # set selected square to be empty
        self.board[row_1][col_1] = selected_piece       # set target square to contain the selected piece
        self.history.append(self.board)                 # append new state of the board to the history

        # increment turns and switch players
        self.turn += 1                                  
        self.switch_player()

    def opponent_color(self):
        """Returns opponents color as a string ("b" or "w")"""
        if self.player == "b":
            return "w"
        else:
            return "b"
        
    def pawn_allowed_moves(self, selected_pawn, board_dimensions = 8):
        """Given a selected pawn, get available moves on board
        selected_pawn = [row, column]
        Returns  [allowed_moves]
        TODO: ADD EN PASSANT"""
        allowed_moves = [] # initialize list of allowed_moves to be returned
        row_index, col_index = selected_pawn # get row index or column index of selected pawn
       
        # first case is player is white
        if self.player == "w":
            # forward check
            if row_index > 0: # square isn't off the board
                if self.board[row_index - 1][col_index] == "--": # no piece in front of pawn
                    allowed_moves.append([row_index - 1, col_index])
                
                # diagonals check
                if col_index >= 0: 
                    if self.board[row_index - 1][col_index-1][0] == "b":        # diagonal left has black piece
                        allowed_moves.append([row_index - 1, col_index-1])
                if col_index < board_dimensions - 1: 
                    if self.board[row_index - 1][col_index+1][0] == "b":        # diagonal rightt has black piece
                        allowed_moves.append([row_index - 1, col_index + 1])

            # two step check 
            if row_index == 6: # white pawn is second rank
                if self.board[row_index - 1][col_index] == "--": # no piece directly in front
                    if self.board[row_index - 2][col_index] == "--": # no piece two steps in front
                        allowed_moves.append([row_index-2, col_index]) # only need to add the two steps move
            

        # second case is player is black
        if self.player == "b":
            # forward check
            if row_index < board_dimensions: # square isn't off the board
                if self.board[row_index + 1][col_index] == "--": # no piece in front of pawn
                    allowed_moves.append([row_index + 1, col_index])
                
                # diagonals check
                if col_index >= 0: 
                    if self.board[row_index + 1][col_index-1][0] == "w":        # diagonal left has white piece
                        allowed_moves.append([row_index + 1, col_index-1])
                if col_index < board_dimensions - 1: 
                    if self.board[row_index + 1][col_index+1][0] == "w":        # diagonal right has white piece
                        allowed_moves.append([row_index + 1, col_index + 1])

            # two step check 
            if row_index == 1: # black pawn is seventh
                if self.board[row_index + 1][col_index] == "--": # no piece directly in front
                    if self.board[row_index + 2][col_index] == "--": # no piece two steps in front
                        allowed_moves.append([row_index + 2, col_index]) # only need to add the two steps move

        return allowed_moves

    def rook_allowed_moves(self, selected_rook, board_dimensions = 8):
        """Given a selected rook get available moves on board.
        selected_rook = [row, column]
        Returns  [allowed_moves]"""
        allowed_moves = []
        row_index, col_index = selected_rook

        # check north direction
        proposed_row = row_index    # default proposed row to the selected rook row
        proposed_col = col_index    # default proposed col to the selected rook col
        for _ in range(row_index):  # make sure on game board
            proposed_row -= 1       # move north one step

            # check if square is empty : 
            if self.board[proposed_row][proposed_col] == "--":   
                allowed_moves.append([proposed_row, proposed_col])
            
            # check if opponents piece
            elif self.board[proposed_row][proposed_col][0] != self.player:
                allowed_moves.append([proposed_row, proposed_col])
                break
            else:
                break

        # check south direction
        proposed_row = row_index    # default proposed row to the selected rook row
        proposed_col = col_index    # default proposed col to the selected rook col
        for _ in range(board_dimensions - row_index - 1):   # make sure on game board
            proposed_row += 1                               # move south one step

            # check if square is empty : 
            if self.board[proposed_row][proposed_col] == "--":   
                allowed_moves.append([proposed_row, proposed_col])
            
            # check if opponents piece
            elif self.board[proposed_row][proposed_col][0] != self.player:
                allowed_moves.append([proposed_row, proposed_col])
                break
            else:
                break

        # check east direction
        proposed_row = row_index    # default proposed row to the selected rook row
        proposed_col = col_index    # default proposed col to the selected rook col
        for _ in range(board_dimensions - col_index - 1):   # make sure on game board
            proposed_col += 1                               # move east one step

            # check if square is empty : 
            if self.board[proposed_row][proposed_col] == "--":   
                allowed_moves.append([proposed_row, proposed_col])
            
            # check if opponents piece
            elif self.board[proposed_row][proposed_col][0] != self.player:
                allowed_moves.append([proposed_row, proposed_col])
                break
            else:
                break

        # check west direction
        proposed_row = row_index    # default proposed row to the selected rook row
        proposed_col = col_index    # default proposed col to the selected rook col
        for _ in range(col_index):  # make sure on game board
            proposed_col -= 1       # move west one step

            # check if square is empty : 
            if self.board[proposed_row][proposed_col] == "--":   
                allowed_moves.append([proposed_row, proposed_col])
            
            # check if opponents piece
            elif self.board[proposed_row][proposed_col][0] != self.player:
                allowed_moves.append([proposed_row, proposed_col])
                break
            else:
                break
        return allowed_moves

    def king_allowed_moves(self, selected_king, board_dimensions = 8):
        """Given a selected king, get available moves on board.
        selected_pawn = [row, column]
        Returns  [allowed_moves]
        TODO make more readable 
        """
        allowed_moves = []                      # initialize list of allowed_moves to be returned
        row_index, col_index = selected_king    # get row index and column index of selected king

        # check adjacent squares
        if row_index + 1 < board_dimensions: 
            if self.board[row_index + 1][col_index][0] != self.player: 
                if not self.is_attacked([row_index + 1, col_index]):
                    allowed_moves.append([row_index + 1, col_index])        # South = [1, 0]
        if row_index - 1 >= 0:
            if self.board[row_index - 1][col_index][0] != self.player: 
                if not self.is_attacked([row_index - 1, col_index]):               
                    allowed_moves.append([row_index - 1, col_index])        # North = [-1, 0]
        if col_index + 1 < board_dimensions:
            if self.board[row_index][col_index + 1][0] != self.player: 
                if not self.is_attacked([row_index, col_index + 1]): 
                    allowed_moves.append([row_index, col_index + 1])        # East = [0, 1]
        if row_index - 1 >= 0:
            if self.board[row_index][col_index - 1][0] != self.player: 
                if not self.is_attacked([row_index, col_index - 1]):               
                    allowed_moves.append([row_index, col_index - 1])        # West = [0, -1]

        # check diagonals
        if row_index + 1 < board_dimensions: 
            if col_index + 1 < board_dimensions:
                if self.board[row_index + 1][col_index + 1][0] != self.player: 
                    if not self.is_attacked([row_index + 1, col_index + 1]):
                        allowed_moves.append([row_index + 1, col_index + 1]) # South-east = [1, 1]
        if row_index + 1 < board_dimensions: 
            if col_index - 1 >= 0:
                if self.board[row_index + 1][col_index - 1][0] != self.player: 
                    if not self.is_attacked([row_index + 1, col_index - 1]):
                        allowed_moves.append([row_index + 1, col_index - 1]) # South-west = [1, -1]
        if row_index - 1 >= 0:               
            if col_index + 1 < board_dimensions:
                if self.board[row_index - 1][col_index + 1][0] != self.player: 
                    if not self.is_attacked([row_index - 1, col_index + 1]):
                        allowed_moves.append([row_index - 1, col_index + 1]) # North-east = [-1, 1]
        if row_index - 1 >= 0:              
            if col_index - 1 >= 0:
                if self.board[row_index - 1][col_index - 1][0] != self.player: 
                    if not self.is_attacked([row_index - 1, col_index - 1]):
                        allowed_moves.append([row_index - 1, col_index - 1]) # North-west = [-1, -1]

        return allowed_moves
        
    def bishop_allowed_moves(self, selected_bishop, board_dimensions = 8):
        """Given a selected bishop, get available moves on board.
        selected_bishop = [row, column]
        Returns  [allowed_moves]
        """
        allowed_moves = []
        row, col = selected_bishop
        new_row = row
        new_col = col

        # southeast diagonals
        SE_range = min(board_dimensions - row - 1, 
                       board_dimensions - col - 1) # check diagonal distance to edge of board
        new_row = row
        new_col = col
        for _ in range(SE_range):   # stay on game board
            new_row += 1            # move south one step
            new_col += 1            # move east one step

            # check if square is empty
            if self.board[new_row][new_col] == "--":   
                allowed_moves.append([new_row, new_col])
            
            # check if opponents piece
            elif self.board[new_row][new_col][0] != self.player:
                allowed_moves.append([new_row, new_col])
                break
            else:
                break

        # southwest diagonals
        SW_range = min(board_dimensions - row - 1, col) # check diagonal distance to edge of board
        new_row = row
        new_col = col
        for _ in range(SW_range):   # stay on game board
            new_row += 1            # move south one step
            new_col -= 1            # move west one step

            # check if square is empty : 
            if self.board[new_row][new_col] == "--":   
                allowed_moves.append([new_row, new_col])
            
            # check if opponents piece
            elif self.board[new_row][new_col][0] != self.player:
                allowed_moves.append([new_row, new_col])
                break
            else:
                break

        # northwest diagonals
        NW_range = min(row, col) # check diagonal distance to edge of board
        new_row = row
        new_col = col
        for _ in range(NW_range):   # stay on game board
            new_row -= 1            # move north one step
            new_col -= 1            # move west one step

            # check if square is empty : 
            if self.board[new_row][new_col] == "--":   
                allowed_moves.append([new_row, new_col])
            
            # check if opponents piece
            elif self.board[new_row][new_col][0] != self.player:
                allowed_moves.append([new_row, new_col])
                break
            else:
                break

        # northeast diagonals
        NE_range = min(row, board_dimensions - col - 1) # check diagonal distance to edge of board
        new_row = row
        new_col = col
        for _ in range(NE_range):   # stay on game board
            new_row -= 1            # move south one step
            new_col += 1            # move east one step

            # check if square is empty : 
            if self.board[new_row][new_col] == "--":   
                allowed_moves.append([new_row, new_col])
            
            # check if opponents piece
            elif self.board[new_row][new_col][0] != self.player:
                allowed_moves.append([new_row, new_col])
                break
            else:
                break
        
        return allowed_moves

    def queen_allowed_moves(self, selected_queen):
        """Given a selected queen, get available moves on board.
        selected_queen = [row, column]
        Returns  [allowed_moves]
        """
        # the allowed moves of a queen are just those of a rook and bishop
        allowed_moves = self.rook_allowed_moves(selected_queen)
        allowed_moves.extend(self.bishop_allowed_moves(selected_queen))
        return allowed_moves

    def knight_allowed_moves(self, selected_knight, board_dimensions = 8):
        """Given a selected knight, get available moves on board.
        selected_knight = [row, column]
        Returns  [allowed_move[0], ...]
        """
        allowed_moves = []
        row, col = selected_knight

        # create relative jump directions
        # see knight_jumps.py for simple code generation
        jumps = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]

        for jump in jumps:
            # create new jump coordinate
            new_row = jump[0] + row
            new_col = jump[1] + col

            # make sure jump is on board
            if new_row in range(board_dimensions):
                if new_col in range(board_dimensions):
                    # allowed squares do not contain your own pieces (or are empty)
                    if self.board[new_row][new_col][0] != self.player:
                        allowed_moves.append([new_row, new_col])
        return allowed_moves

    def is_valid_move(self, move_pair):
        """"This function checks if a move from move_pair = [position_0, position_1]
        position_0 = [row_0, col_0] to position_1 = [row_1, col_1] is valid. 
        Returns True if valid move."""
        row_0, col_0 = move_pair[0]
        color_selected = self.board[row_0][col_0][0]    # get "w" from "wP" for example

        # check if position_0 has a piece 
        if self.board[row_0][col_0] == "--":
            return False
        
        # check if position_0 has correct color for turn order
        if color_selected != self.player:  
            return False
        
        # if piece is pawn, check allowed moves
        if self.board[row_0][col_0][1] == "P":
            allowed_moves = self.pawn_allowed_moves(move_pair[0])
            if move_pair[1] in allowed_moves:
                return True
        
        # if piece is rook, check allowed moves
        if self.board[row_0][col_0][1] == "R":
            allowed_moves = self.rook_allowed_moves(move_pair[0])
            if move_pair[1] in allowed_moves:
                return True
            
        # if piece is king, check allowed moves
        if self.board[row_0][col_0][1] == "K":
            allowed_moves = self.king_allowed_moves(move_pair[0])
            if move_pair[1] in allowed_moves:
                return True
            
        # if piece is bishop, check allowed moves
        if self.board[row_0][col_0][1] == "B":
            allowed_moves = self.bishop_allowed_moves(move_pair[0])
            if move_pair[1] in allowed_moves:
                return True
        
        # if piece is queen, check allowed moves
        if self.board[row_0][col_0][1] == "Q":
            allowed_moves = self.queen_allowed_moves(move_pair[0])
            if move_pair[1] in allowed_moves:
                return True
        
        # if piece is knight, check allowed moves
        if self.board[row_0][col_0][1] == "N":
            allowed_moves = self.knight_allowed_moves(move_pair[0])
            if move_pair[1] in allowed_moves:
                return True
        
        else:
            return False
     
    def get_valid_moves(self, coord):
        """Return the valid possible moves of the coordinates as a list of coordinates [[row, col], ...]. If the selected_piece is the empty square"""
        FUNCTION_DICT = {"P": "pawn_allowed_moves", "B": "bishop_allowed_moves", 
                         "Q": "queen_allowed_moves", "R": "rook_allowed_moves",
                         "K": "king_allowed_moves", "N": "knight_allowed_moves"}
        valid_moves = []
        row, col = coord
        piece_color = self.board[row][col][0]
        piece_name = self.board[row][col][1]
        if piece_name != "-":
            if piece_color == self.player:
                valid_move_func = getattr(GameState, FUNCTION_DICT[piece_name])
                valid_moves = valid_move_func(self, coord)
        return valid_moves

    """After this point are things to be done"""

    def is_attacked(self, coord):
        """Check if square at coords = [row, col] is being attacked by opponent.
        Returns true if square is being attacked. 
        TODO implement cases where is True"""
        return False
    
    def is_checkmate(self):
        """Write code using is_attacked and king_valid_move.
        Return true if current board position is checkmate"""
        pass

    def is_stalemate(self):
        """Checks in history to determine if current position is stalemate."""
        pass





        


    