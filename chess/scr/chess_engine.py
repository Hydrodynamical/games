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
        pass
    
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
        self.history.append(self.board)  # append the new state of the board to the history

        # increment turns and switch players
        self.turn += 1                                  
        self.switch_player()

    def is_valid_move(self, move_pair):
        """"This function checks if a move from move_pair = [position_0, position_1]
        position_0 = [row_0, col_0] to position_1 = [row_1, col_1] is valid. 
        Returns True if valid move, and False if not valid move"""
        row_0, col_0 = move_pair[0]
        color_selected = self.board[row_0][col_0][0]    # get "w" from "wP" for example

        # check if position_0 has a piece 
        if self.board[row_0][col_0] == "--":
            return False
        
        # check if position_0 has correct color for turn order
        if color_selected != self.player:   # this returns False when position_0 is empty
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
        
        else:
            return False
        
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
        """Given a selected rook get available moves on board
        selected_rook = [row, column]
        Returns  [allowed_moves]"""
        allowed_moves = []
        row_index, col_index = selected_rook

        # 0 <= row_index + incr < board_dimensions
        row_incr_range_neg = range(-row_index, 0, -1) # negative allowed row increments
        row_incr_range_pos = range(1, board_dimensions - row_index) # don't allow trivial move

        # 0 <= col_index + incr < board_dimensions
        col_incr_range_neg = range(-col_index, 0, -1) # negative allowed column increments 
        col_incr_range_pos = range(1, board_dimensions - col_index) # don't allow trivial move
        
        # up direction 
        for row_incr in row_incr_range_neg:
            # check if piece of any color is on the proposed square
            if self.board[row_index + row_incr][col_index] == "--":
                allowed_moves.append([row_index + row_incr, col_index])
            else:
                break
       
        # down direction
        for row_incr in row_incr_range_pos:
            # check if piece of any color is on the proposed square
            if self.board[row_index + row_incr][col_index] == "--":
                allowed_moves.append([row_index + row_incr, col_index])
            else:
                break
        """
        # left direction
        for col_incr in col_incr_range_neg:
            # check if piece of any color is on the proposed square
            if self.board[row_index][col_index + col_incr] == "--":
                allowed_moves.append([row_index, col_index + col_incr])
            else:
                break
        
        # right direction
        for col_incr in col_incr_range_pos:
            # check if piece of any color is on the proposed square
            if self.board[row_index][col_index + col_incr] == "--":
                allowed_moves.append([row_index, col_index + col_incr])
            else:
                break
        """
        print(allowed_moves)
        return allowed_moves







        


    