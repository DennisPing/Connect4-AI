# Connect4-AI

Connect4 game with an AI opponent

CS-5100: Fundamentals of Artifical Intelligence  
Dennis Ping  
Madhu Patar  

## Requirements

```txt
python 3
numpy
beautifulsoup4
numba
selenium
```

## Chromedriver

The AI needs a Google Chrome driver in order to control the web browser.

Please [download a Chromedriver](https://chromedriver.chromium.org/downloads) and place it in this current folder. MacOS, Windows, and Linux all use different Chromedrivers.

## How to Run

```txt
python connect4-fast.py
```

## Online Mode

This AI plays online on the website: https://connect-4.org

The human always needs to create the game room, since the AI doesn't know how to create new rooms. Then, the human gives the AI the room code which lets the AI join the game. Wait a few seconds for the AI to load the browser, parse the HTML elements, and it's ready to play automatically!

```txt
Enter the 4 digit game room: 5432
```

## AI Algorithms

- [ ] Minimax Algorithm  
- [x] Minimax with Alpha Beta Pruning Algorithm  
- [ ] Monte Carlo Algorithm  
