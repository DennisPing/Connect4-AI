import os
import numpy as np
from time import time
from game_state import State
from minimax_alphabeta import alphabeta_search
from minimax import basic_minimax
from colorama import Fore

class Game:
    AI = -1
    PLAYER = 0

    def __init__(self):
        self.current_state = State(0, 0)
        self.turn = self.PLAYER
        self.first = self.turn
        self.rounds = 0
        self.depth = 7
        self.node_count = 0
        self.compute_time = 0

    def is_game_over(self):
        if self.has_winning_state():
            """Display who won"""
            print("AI Bot won!") if ~self.turn == self.AI else print("Congratulations, you won!")
            return True
        elif self.draw():
            print("Draw. No winner.")
            return True
        return False

    def draw(self):
        """Check current state to determine if it is in a draw"""
        return self.current_state.is_draw(self.current_state.game_bitboard) and not self.has_winning_state()

    def has_winning_state(self):
        return self.current_state.four_in_a_row(self.current_state.ai_bitboard) or self.current_state.four_in_a_row(
            self.current_state.human_bitboard)

    def next_turn(self):
        if self.turn == self.AI:
            self.rounds += 1
            if self.rounds % 5 == 0 and self.depth < 10:
                self.depth += 1
            self.query_AI(self.depth)
        else:
            self.query_player()
        self.turn = ~self.turn

    def query_player(self):
        """Make a move by querying standard input."""
        column = None
        while column is None:
            try:
                column = input("Column number from 1 to 7: ")
                column = int(column) - 1 # Humans read from 1-7 but computers read from 0-6
                # Check if move is legal
                if not 0 <= column <= 6:
                    raise ValueError
                if self.current_state.game_bitboard & (1 << (7 * column + 5)):
                    raise IndexError
            except (ValueError, IndexError):
                print("Invalid move. Try again...")
                column = None

        _, new_game_bitboard = self.current_state.make_move(self.current_state.human_bitboard,
                                                    self.current_state.game_bitboard, column)
        self.current_state = State(self.current_state.ai_bitboard, new_game_bitboard, self.current_state.depth + 1)

    def query_AI(self, depth):
        """ AI Bot chooses next best move from current state """
        t1 = time()
        self.current_state, node_count = alphabeta_search(self.current_state, self.first, d=depth)
        self.compute_time = round(time() - t1, 2)
        # self.current_state, node_count = basic_minimax(self.current_state, self.first, d=depth)
        self.node_count = node_count

    def pretty_print_board(self, gridboard):

        #clear console/terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Game rounds: {self.rounds}")
        print(f"AI search depth: {self.depth}")
        print(f"Nodes searched: {self.node_count}")
        print(f"Compute time: {self.compute_time} sec")
        print('')
        print("\t      1   2   3   4   5   6   7 ")
        print("\t      -   -   -   -   -   -   - ")

        for r in range(gridboard.shape[0]):
            print("\t", 6-r,' ', end="")
            for c in range(gridboard.shape[1]):
                if gridboard[r, c] == 2:
                    print("| " + Fore.BLUE + 'x' + Fore.RESET, end=" ")   #print colored 'x'
                elif gridboard[r, c] == 1:
                    print("| " + Fore.RED + 'o' + Fore.RESET, end=" ")   #print colored 'o'
                else:
                    print("| " + ' ', end=" ")

            print("|")
        print('')

    def bitboard_to_array(self):
        """
        Helper method to pretty print binary board (6x7 board with top sentinel row of 0's)
        """
        ai_board, total_board = self.current_state.ai_bitboard, self.current_state.game_bitboard
        output = np.zeros((6,7), dtype=np.int64)
        for row in range(5, -1, -1):
            for col in range(0, 7):
                if ai_board & (1 << (7 * col + row)):
                    output[row, col] = 1
                elif total_board & (1 << (7 * col + row)):
                    output[row, col] = 2
                else:
                    output[row, col] = 0
        return np.flipud(output)

if __name__ == "__main__":
    print("Welcome to Connect Four!")

    while True:
        game = Game()
        while not game.is_game_over():
            game.next_turn()
            if game.turn == 0:
                gridboard = game.bitboard_to_array()
                game.pretty_print_board(gridboard)
        break