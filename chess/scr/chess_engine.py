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
        row_0, col_0 = move_pair[0]
        row_1, col_1 = move_pair[1]
        selected_piece = self.board[row_0][col_0]
        self.board[row_0][col_0] = "--"
        self.board[row_1][col_1] = selected_piece
        self.turn += 1
        self.switch_player()

    def is_valid_move(self, move_pair):
        """"This function checks if a move from move_pair = [position_0, position_1]
        position_0 = [row_0, col_0] to position_1 = [row_1, col_1] is valid. 
        Returns True if valid move, and False if not valid move"""
        row_0, col_0 = move_pair[0]
        row_1, col_1= move_pair[1]
        color_selected = self.board[row_0][col_0][0]    # get "w" from "wP" for example

        # check if position_0 has a piece 
        if self.board[row_0][col_0] == "--":
            return False
        
            # check if position_0 has correct color for turn order
        elif color_selected != self.player:   # this returns False when position_0 is empty
            return False
        else:
            return True
        


    