import numpy as np
import sys
import time
from typing import List, Tuple
from numba import njit, typed
from base_game import create_board, is_valid_column, get_valid_columns, \
    get_next_open_row, drop_piece, check_for_win, pretty_print_board, \
    evaluate_position

# The main file for playing minimax basic AI.

@njit
def minimax_basic(board:np.ndarray, depth:int, maxTurn:bool, node_count:int) -> Tuple[int, int, int]:
    """ Implementation of Minimax Alpha Beta Pruning

    Args:
        board (np.ndarray): game board
        depth (int): recursive search depth
        maxTurn (bool): True if it's max turn; False if it's min turn
        node_count (int): The accumulator node_count

    Returns:
        Tuple[int, int, int]: best_column, best_score, node_count
    """

    PLAYER_PIECE = 1
    AI_PIECE = 2
    WIN_SCORE = 100000
    TIE = -1
    AGING_PENALTY = 3

    # First, check for possible winning move -> Stop recursing and return the minimax_alphabeta score
    if check_for_win(board, PLAYER_PIECE):
        score = WIN_SCORE + depth * AGING_PENALTY # If there are multiple win possibilites, choose the faster one.
        return typed.List([0, score, node_count])

    if check_for_win(board, AI_PIECE):
        score = -WIN_SCORE - depth * AGING_PENALTY # If there are multiple win possibilites, choose the faster one.
        return typed.List([0, score, node_count])

    # If the board is full -> return tie
    if len(get_valid_columns(board)) == 0:
        return typed.List([0, TIE, node_count])

    # If search depth == 0 -> Stop recursing and return the minimax_alphabeta score
    if depth == 0:
        bestCol = get_valid_columns(board)[0]
        return typed.List([bestCol, evaluate_position(board, AI_PIECE), node_count])
    
    if maxTurn:
        value = -sys.maxsize
        bestCol = 0
        for col in get_valid_columns(board):
            row = get_next_open_row(board, col)
            temp_board = np.copy(board)
            temp_board[row, col] = PLAYER_PIECE
            _, score, node_count = minimax_basic(temp_board, depth - 1, False, node_count)
            
            if score > value:
                value = score
                bestCol = col
        return typed.List([bestCol, value, node_count + 1])
    
    else:
        value = sys.maxsize
        bestCol = 0
        for col in get_valid_columns(board):
            row = get_next_open_row(board, col)
            temp_board = np.copy(board)
            temp_board[row, col] = AI_PIECE
            _, score, node_count = minimax_basic(temp_board, depth - 1, True, node_count)
    
            if score < value:
                value = score
                bestCol = col
        return typed.List([bestCol, value, node_count + 1])

def start_game():
    """ 
    Initialize the game and play until the game is over.
    Human player goes first.
    Starting search depth = 7 and increases on every 5th round.
    Search depth is capped at 10.
    """

    PLAYER_PIECE = 1
    AI_PIECE = 2
    HUMAN_TURN = True

    board = create_board()
    game_over = False
    rounds = 0
    depth = 7
    computation_time = 0

    while not game_over:
        endgame = ''
        if HUMAN_TURN:
            col = int(input("Player 1 make your selection (1-7): "))
            col = col - 1 # Humans read from 1-7 but computers read from base 0 (0-6)
            if is_valid_column(board, col):
                row = get_next_open_row(board, col) 
                board = drop_piece(board, row, col, PLAYER_PIECE)

                if check_for_win(board, PLAYER_PIECE):
                    endgame = "Player 1 wins!"
                    game_over = True

        if not HUMAN_TURN:
            rounds += 1
            t1 = time.time()
            if rounds % 5 == 0 and depth < 10: # Cap the search depth at 10.
                depth += 1
            col, score, node_count = minimax_basic(board, depth, True, 0)
            computation_time = round(time.time() - t1, 2)
            if score == -1: # Check for tie
                endgame = "Tie!"
                game_over = True
            row = get_next_open_row(board, col)
            board = drop_piece(board, row, col, AI_PIECE)

            if check_for_win(board, AI_PIECE):
                endgame = "Player 2 wins!"
                game_over = True
            pretty_print_board(np.flipud(board), rounds, depth, node_count, computation_time, endgame)

        HUMAN_TURN = not HUMAN_TURN

    pretty_print_board(np.flipud(board), rounds, depth, node_count, computation_time, endgame)

if __name__ == "__main__":
    start_game()