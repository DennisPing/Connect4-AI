# Connect4-AI

**CS-5100: Fundamentals of Artifical Intelligence**  
**Dennis Ping**  
**Madhu Patar**  

Play the game Connect4 against an AI opponent ft. offline and online mode.

🤖 AI <---> local terminal <---> 👨‍💻 Human Player

🤖 AI <---> connect-4.org <---> 👩‍💻 Human Player

## Requirements

```txt
python 3
numpy
beautifulsoup4
numba
selenium
colorama
```

## How It Works

The AI can use 3 different algorithms to play Connect 4:

  1. Minimax
  2. Minimax Alpha Beta Pruning
  3. Monte Carlo Search  

The AI needs to be initiated by a human (*the AI is not sentient... yet*). The AI loads up a game instance on the website connect-4.org and gives back the game URL to the human handler. The human handler can then give the URL to anybody so they can against play the AI online.

## How to Run

### ✅ Install Requirements

```txt
python -m pip install -r requirements.txt
```

### 1️⃣ Basic Minimax

```txt
python play_minimax_basic.py
```

### 2️⃣ Minimax Alpha Beta Pruning

```txt
python play_minimax_alphabeta.py
```

### 3️⃣ Monte Carlo Tree Search

```txt
python play_montecarlo.py
```

### 4️⃣ Minimax Alpha Beta Pruning using Bitboard

```txt
python play_bitboard.py
```

## Project Objectives

Investigate the effeciency and performance of each algorithm my recording the number of nodes the algorithm searches through before determining its "best move" to play.

## Requirements for the Human Handler: Chromedriver

The AI needs a [Google chromedriver](https://chromedriver.chromium.org/downloads) in order to control the web browser.

The `online-hander.py` script uses the appropriate chromedriver for Windows, Mac, or Linux. At the time of writing, the Google Chrome version is `Chrome version 92`. If you are using a newer Chrome version, please overwrite the existing chromedriver in the folder `/chromedrivers`.

## AI Algorithms

- [x] Minimax Algorithm  
- [x] Minimax with Alpha Beta Pruning Algorithm  
- [ ] Monte Carlo Algorithm  
