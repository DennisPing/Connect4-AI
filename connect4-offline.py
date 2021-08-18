import numpy as np
import sys
import time
from typing import List, Tuple
from numba import njit, typed

def create_board() -> np.ndarray:
    return np.zeros((6, 7))

def drop_piece(board:np.ndarray, row:int, col:int, piece:int) -> np.ndarray:
    """ Place the piece on the board at coordinates [row, col] """
    board[row, col] = piece
    return board

@njit
def is_valid_column(board:np.ndarray, col:int) -> bool:
    """ Check if this is column is not full """
    return board[6 - 1, col] == 0

@njit
def get_valid_columns(board:np.ndarray) -> List[int]:
    """ Get a list of all non-full columns """
    valid_locations = []
    for col in typed.List([3,4,2,5,1,6,0]): # Favor middle columns to be better
        if is_valid_column(board, col):
            valid_locations.append(col)
    return typed.List(valid_locations)

@njit
def get_next_open_row(board:np.ndarray, col:int) -> int:
    """ Get the next open row index in this column """
    for row in range(6):
        if board[row, col] == 0:
            return row
    return -1

# @njit
# def print_board( board:np.ndarray) -> str:
#     print(np.flipup(board))
#     print()

@njit
def check_for_win(board:np.ndarray, piece:int) -> bool:
    """ Check for 4 in a row with brute force """
    # check horizontal locations for a win
    for r in range(6):
        for c in range(7 - 3):
            if board[r, c] == piece and board[r, c + 1] == piece and board[r, c + 2] == piece and board[r, c + 3] == piece:
                return True
    
    # check vertical locations for a win
    for r in range(6 - 3):
        for c in range(7):
            if board[r, c] == piece and board[r + 1, c] == piece and board[r + 2, c] == piece and board[r + 3, c] == piece:
                return True

    # check positively sloped diagonals
    for r in range(6 - 3):        
        for c in range(7 - 3):
            if board[r, c] == piece and board[r + 1, c + 1] == piece and board[r + 2, c + 2] == piece and board[r + 3, c + 3] == piece:
                return True

    # check negatively sloped diagonals
    for r in range(3, 6):
        for c in range(7 - 3):
            if board[r, c] == piece and board[r - 1, c + 1] == piece and board[r - 2, c + 2] == piece and board[r - 3, c + 3] == piece:
                return True
    return False

@njit
def evaluate_position(board:np.ndarray, piece:int) -> int:
    """ Calculate the score of the current board state by brute force

    Args:
        board (np.ndarray): game board
        piece (int): the player being scored (human or AI)

    Returns:
        int: the score of the current board state
    """
    score = 0

    # Score horizontal
    for r in range(6):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(7 - 3):
            window = row_array[c: c + 4]
            score += score_window(window, piece)

    # Score vertical
    for c in range(7):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(6 - 3):
            window = col_array[r: r + 4]
            score += score_window(window, piece)

    # Score positive diagonal
    for r in range(6 - 3):
        for c in range(7 - 3):
            window = [board[r + i, c + i] for i in range(4)]
            score += score_window(window, piece)

    # Score negative diagonal
    for r in range(6 - 3):
        for c in range(7 - 3):
            window = [board[r + 3 - i, c + i] for i in range(4)]
            score += score_window(window, piece)
    return score

@njit
def score_window(window:List[int], piece:int) -> int:
    """ Quantify a 4 block window

    Args:
        window (List[int]): [description]
        piece (int): [description]

    Returns:
        int: [description]
    """
    score = 0
    opponent_piece = piece % 2 + 1

    num_offense = window.count(piece)
    num_defense = window.count(opponent_piece)
    num_empty = window.count(0)

    if num_offense == 4:
        score += 100
    elif num_offense == 3 and num_empty == 1:
        score += 24
    elif num_offense == 2 and num_empty == 2:
        score += 12
    
    if num_defense == 4:
        score -= 100
    elif num_defense == 3 and num_empty == 1:
        score -= 12
    elif num_defense == 2 and num_empty == 2:
        score -= 6
    return score 

# @njit(types.UniTuple(types.int64, 2)(types.int64[::1], types.int64, types.int64, types.int64, types.boolean))
@njit
def minimax(board:np.ndarray, depth:int, alpha:int, beta:int, maxTurn:bool) -> Tuple[int, int]:
    """ Implementation of Minimax Alpha Beta Pruning

    Args:
        board (np.ndarray): game board
        depth (int): recursive search depth
        alpha (int): initialized as negative infinity
        beta (int): initialized as positive infinity
        maxTurn (bool): True if it's max turn; False if it's min turn

    Returns:
        Tuple[int, int]: best_column, best_score
    """

    PLAYER_PIECE = 1
    AI_PIECE = 2
    WIN_SCORE = 100000
    TIE = -1
    AGING_PENALTY = 3

    # First, check for possible winning move -> Stop recursing and return the minimax score
    if check_for_win(board, PLAYER_PIECE):
        score = WIN_SCORE + depth * AGING_PENALTY # If there are multiple win possibilites, choose the faster one.
        return typed.List([0, score])

    if check_for_win(board, AI_PIECE):
        score = -WIN_SCORE - depth * AGING_PENALTY # If there are multiple win possibilites, choose the faster one.
        return typed.List([0, score])

    # If the board is full -> return tie
    if len(get_valid_columns(board)) == 0:
        return typed.List([0, TIE])

    # If search depth == 0 -> Stop recursing and return the minimax score
    if depth == 0:
        return typed.List([0, evaluate_position(board, AI_PIECE)])
    
    if maxTurn:
        value = -sys.maxsize
        bestCol = 0
        for col in get_valid_columns(board):
            row = get_next_open_row(board, col)
            temp_board = np.copy(board)
            temp_board[row, col] = PLAYER_PIECE
            score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
            
            if score > value:
                value = score
                bestCol = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return typed.List([bestCol, value])
    
    else:
        value = sys.maxsize
        bestCol = 0
        for col in get_valid_columns(board):
            row = get_next_open_row(board, col)
            temp_board = np.copy(board)
            temp_board[row, col] = AI_PIECE
            score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
    
            if score < value:
                value = score
                bestCol = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return typed.List([bestCol, value])

def start_game():
    """ Initialize the game and play the game until the game is over.
    """

    PLAYER_PIECE = 1
    AI_PIECE = 2
    HUMAN_TURN = True

    board = create_board()
    game_over = False

    while not game_over:
        
        if HUMAN_TURN:
            col = int(input("Player 1 make your selection (1-7): "))
            col = col - 1 # Humans read from 1-7 but computers read from base 0 (0-6)
            if is_valid_column(board, col):
                row = get_next_open_row(board, col) 
                board = drop_piece(board, row, col, PLAYER_PIECE)

                if check_for_win(board, PLAYER_PIECE):
                    print("Player 1 wins!")
                    game_over = True

        if not HUMAN_TURN:
            t1 = time.time()
            col = minimax(board, 8, -sys.maxsize, sys.maxsize, True)[0]
            print(round(time.time() - t1), 3)
            row = get_next_open_row(board, col)
            board = drop_piece(board, row, col, AI_PIECE)

            if check_for_win(board, AI_PIECE):
                print("Player 2 wins!")
                game_over = True
            print(np.flipud(board))

        HUMAN_TURN = not HUMAN_TURN

    print(np.flipud(board)) # Print the final board

if __name__ == "__main__":
    start_game()