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
        board_length = 8

        # print board
        for row in range(board_length):
            row_string = str(self.board[row])
            print(row_string + "\n")
        
        # print current player
        print("Current player is: " + self.player)

        # print turn number
        print("Turn number = " + str(self.turn))

        # asthetic 
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

    