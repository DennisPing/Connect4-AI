import sys

def basic_minimax(state, turn=-1, d=7):
    """Search game state to determine best action; use alpha-beta pruning. """

    # Functions used by alpha beta
    def max_value(state, depth, cnt):
        if cutoff_search(state, depth):
            return state.calculate_heuristic(), cnt + 1

        v = -sys.maxsize
        for child in state.generate_children(turn):
            temp_v, cnt = min_value(child, depth + 1, cnt)
            v = max(v, temp_v)
        if v == -sys.maxsize:
            # If win/loss/draw not found, don't return -infinity to MIN node
            return sys.maxsize, cnt + 1
        return v, cnt + 1

    def min_value(state, depth, cnt):
        if cutoff_search(state, depth):
            return state.calculate_heuristic(), cnt + 1

        v = sys.maxsize
        for child in state.generate_children(turn):
            if child in seen:
                continue
            temp_v, cnt = max_value(child, depth + 1, cnt)
            v = min(v, temp_v)
        if v == sys.maxsize:
            # If win/loss/draw not found, don't return infinity to MAX node
            return -sys.maxsize, cnt + 1
        return v, cnt + 1

    # Keep track of seen states using their hash
    seen = {}

    # Body of alpha beta_search:
    cutoff_search = (lambda state, depth: depth > d or state.terminal_node_test())
    best_score = -sys.maxsize
    best_action = None
    for child in state.generate_children(turn):
        v, cnt = min_value(child, 1, 0)
        if v > best_score:
            best_score = v
            best_action = child
    return best_action, cnt