import numpy as np
import sys
import os
import time
from typing import List, Tuple
from numba import njit, typed
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def create_board() -> np.ndarray:
    return np.zeros((6, 7))

@njit
def drop_piece(board:np.ndarray, row:int, col:int, piece:int) -> np.ndarray:
    board[row, col] = piece
    return board

@njit
def is_valid_column(board:np.ndarray, col:int) -> bool:
    return board[board.shape[0] - 1, col] == 0

@njit
def get_valid_columns(board:np.ndarray) -> List[int]:
    valid_locations = []
    for col in typed.List([3,4,2,5,1,6,0]): # Favor middle columns to be better
        if is_valid_column(board, col):
            valid_locations.append(col)
    return typed.List(valid_locations)

@njit
def get_next_open_row(board:np.ndarray, col:int) -> int:
    for row in range(board.shape[0]):
        if board[row, col] == 0:
            return row
    return -1 # This is never actually returned, it's just done to satisfy Numba.

@njit
def check_for_win(board:np.ndarray, piece:int) -> bool:
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
    score = 0
    num_rows = board.shape[0]
    num_cols = board.shape[1]

    # Score horizontal
    for r in range(num_rows):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(7 - 3):
            window = row_array[c: c + 4]
            score += score_window(window, piece)

    # Score vertical
    for c in range(num_cols):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(num_rows - 3):
            window = col_array[r: r + 4]
            score += score_window(window, piece)

    # Score positive diagonal
    for r in range(num_rows - 3):
        for c in range(num_cols - 3):
            window = [board[r + i, c + i] for i in range(4)]
            score += score_window(window, piece)

    # Score negative diagonal
    for r in range(num_rows - 3):
        for c in range(num_cols - 3):
            window = [board[r + 3 - i, c + i] for i in range(4)]
            score += score_window(window, piece)
    return score

@njit
def score_window(window:List[int], piece:int) -> int:
    score = 0
    opponent_piece = piece % 2 + 1

    num_offense = window.count(piece)
    num_defense = window.count(opponent_piece)
    num_empty = window.count(0)

    if num_offense == 4:
        score += 1000
    elif num_offense == 3 and num_empty == 1:
        score += 50
    elif num_offense == 2 and num_empty == 2:
        score += 10
    
    if num_defense == 4:
        score -= 1000
    elif num_defense == 3 and num_empty == 1:
        score -= 100
    elif num_defense == 2 and num_empty == 2:
        score -= 10
    return score 

@njit
def minimax(board:np.ndarray, depth:int, alpha:int, beta:int, maxTurn:bool) -> Tuple[int, int]:

    PLAYER_PIECE = 1
    AI_PIECE = 2
    WIN_SCORE = 1000000
    TIE = -1
    AGING_PENALTY = 3

    # First, check for possible winning move -> Stop recursing and return the minimax score
    if check_for_win(board, PLAYER_PIECE):
        score = WIN_SCORE + depth * AGING_PENALTY # If there are multiple win possibilites, choose the faster one.
        return 0, score

    if check_for_win(board, AI_PIECE):
        score = -WIN_SCORE - depth * AGING_PENALTY # If there are multiple win possibilites, choose the faster one.
        return 0, score

    # If the board is full -> return tie
    if len(get_valid_columns(board)) == 0:
        return 0, TIE

    # If search depth == 0 -> Stop recursing and return the minimax score
    if depth == 0:
        return 0, evaluate_position(board, AI_PIECE)
    
    if maxTurn:
        value = -sys.maxsize
        bestCol = 0
        for col in get_valid_columns(board):
            row = get_next_open_row(board, col)
            temp_board = np.copy(board)
            temp_board = drop_piece(temp_board, row, col, PLAYER_PIECE)
            score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
            if score > value:
                value = score
                bestCol = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return bestCol, value
    
    else:
        value = sys.maxsize
        bestCol = 0
        for col in get_valid_columns(board):
            row = get_next_open_row(board, col)
            temp_board = np.copy(board)
            temp_board = drop_piece(temp_board, row, col, AI_PIECE)
            score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
            if score < value:
                value = score
                bestCol = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return bestCol, value

def open_website(driver, url:str, wait_time:int=5) -> None:
    driver.get(url)
    time.sleep(wait_time) # Need to wait for all dynamic HTML elements to load

def parse_page(driver) -> np.ndarray:
    online_table = np.zeros((6,7))
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    table_rows = soup.find_all("tr", attrs={"class": "ng-star-inserted"})

    for r, row in enumerate(table_rows):
        target_col = row.find_all("td") # Filter by td tag.
        for c, col in enumerate(target_col):
            target = str(col.find("div"))
            if len(target) > 2 and "background-color" in target:
                if "red" in target: # Red is player 1 (human)
                    online_table[r, c] = 1
                if "blue" in target: # Blue is player 2 (AI)
                    online_table[r, c] = 2
            elif len(target) > 2:
                online_table[r, c] = 0
    return online_table

def get_player_turn(driver) -> bool:
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    player_order = soup.find("div", attrs={"class": "current-turn-container"})
    paragraph = player_order.find("p")
    if paragraph.get_text() == "It's your opponent's turn":
        return True # Human goes first
    else:
        return False # AI goes first

def auto_click(driver, row:int, col:int) -> None:
    table = driver.find_element_by_xpath("//table[@class='center-table']")
    target_row = table.find_elements_by_xpath(".//tr")[row]
    target_box = target_row.find_elements_by_xpath(".//td")[col]
    target_box.click()
    print(f"I clicked on [{row}, {col})]")

def check_game_over(driver) -> str:
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    game_over_box = soup.find("app-game-end")
    card_content = game_over_box.find("mat-card-content")
    paragraph = card_content.find("p")
    if paragraph.get_text() == "You Won":
        return "won"
    elif paragraph.get_text() == "You Lost":
        return "lost"
    else:
        return "continue"

def start_game():

    room = input("Enter the 4 digit game room: ")
    url = f"http://connect-4.org/?lb{room}"

    AI_PIECE = 2

    options = Options()
    options.add_argument("start-maximized")
    if not os.path.exists(f"{os.getcwd()}/chromedriver"): # Check if chromedriver exists in current directory
        print("Error, Chromedriver not found!")
        return
    driver = webdriver.Chrome(f"{os.getcwd()}/chromedriver", options=options)
    open_website(driver, url, wait_time=4)
    board = parse_page(driver)

    game_over = False
    rounds = 0
    depth = 5
    HUMAN_TURN = get_player_turn(driver)
    while not game_over:
        rounds += 1
        while HUMAN_TURN:
            HUMAN_TURN = get_player_turn(driver) # Continously check the webpage for player turn
            time.sleep(0.5)
            board = parse_page(driver)
            board = np.flip(board, axis = 0)
            outcome = check_game_over(driver)
            if outcome == "won": # The AI is checking whether it won or lost
                print("Player 2 wins!")
                game_over = True
                break
            elif outcome == "lost": 
                print("Player 1 wins!")
                game_over = True
                break
            else:
                continue

        if not HUMAN_TURN:
            board = parse_page(driver)
            board = np.flip(board, axis = 0)

            if rounds % 2 == 0 and depth < 10: # Cap the search depth at 10.
                depth += 1
            print(f"The search depth is: {depth}")

            t1 = time.time()
            col = minimax(board, depth, -sys.maxsize, sys.maxsize, True)[0]
            print(round(time.time() - t1, 3))
            row = get_next_open_row(board, col)
            
            auto_click(driver, row, col)
            time.sleep(0.1)
            board = parse_page(driver)
            board = np.flip(board, axis = 0)
            if check_for_win(board, AI_PIECE):
                print("Player 2 wins!")
                game_over = True
            HUMAN_TURN = True
    time.sleep(1)
    driver.close()

if __name__ == "__main__":
    start_game()