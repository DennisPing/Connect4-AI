import sys
import numpy as np

class State:

    status = 3

    def __init__(self, ai_bitboard, game_bitboard, depth=0):
        self.ai_bitboard = ai_bitboard
        self.game_bitboard = game_bitboard
        # self.human_bitboard = ai_bitboard ^ game_bitboard
        self.depth = depth

    @property
    def human_bitboard(self):
        return self.ai_bitboard ^ self.game_bitboard

    def four_in_a_row(self, bitboard):
        # Horizontal check
        m = bitboard & (bitboard >> 7)
        if m & (m >> 14):
            return True
        # Diagonal \
        m = bitboard & (bitboard >> 6)
        if m & (m >> 12):
            return True
        # Diagonal /
        m = bitboard & (bitboard >> 8)
        if m & (m >> 16):
            return True
        # Vertical
        m = bitboard & (bitboard >> 1)
        if m & (m >> 2):
            return True
        # Nothing found
        return False

    def is_draw(self, bitboard):
        return all(bitboard & (1 << (7 * column + 5)) for column in range(0, 7))

    def terminal_node_test(self):
        """ Test if current state is a terminal node """
        if self.four_in_a_row(self.ai_bitboard):
            # AI Wins
            self.status = -1
            return True
        elif self.four_in_a_row(self.human_bitboard):
            # Player Wins
            self.status = 1
            return True
        elif self.is_draw(self.game_bitboard):
            # Draw
            self.status = 0
            return True
        else:
            return False

    def calculate_heuristic(self):
        """
        Score based on who can win. Score computed as 22 minus number of moves played
        i.e. AI wins with 4th move, score = 22 - 4 = 18
        """
        if self.status == -1:
            # AI Wins
            return 22 - (self.depth // 2)
        elif self.status == 1:
            # Player Wins
            return -1 * (22 - (self.depth // 2))
        elif self.status == 0:
            # Draw
            return 0
        elif self.depth % 2 == 0:
            # MAX node returns
            return sys.maxsize
        else:
            # MIN node returns
            return -sys.maxsize

    def generate_children(self, who_went_first):
        """ For each column entry, generate a new State if the new position is valid"""
        for i in range(0, 7):
            # Select column starting from the middle and then to the edges index order [3,2,4,1,5,0,6]
            column = 3 + (1 - 2 * (i % 2)) * (i + 1) // 2
            if not self.game_bitboard & (1 << (7 * column + 5)):
                if (who_went_first == -1 and self.depth % 2 == 0) or (who_went_first == 0 and self.depth % 2 == 1):
                    # AI (MAX) Move
                    new_ai_position, new_game_position = self.make_move(self.ai_bitboard, self.game_bitboard, column)
                else:
                    # Player (MIN) move
                    new_ai_position, new_game_position = self.make_move_opponent(self.ai_bitboard, self.game_bitboard,
                                                                            column)
                yield State(new_ai_position, new_game_position, self.depth + 1)

    def __str__(self):
        """ At position 0, format int to binary using 49 digits zero padding. We don't need to use all 64 digits. """
        return '{0:049b}'.format(self.ai_bitboard) + ' ; ' + '{0:049b}'.format(self.game_bitboard)

    def __hash__(self):
        return hash((self.ai_bitboard, self.game_bitboard, self.depth % 2))

    def __eq__(self, other):
        return (self.ai_bitboard, self.game_bitboard, self.depth % 2) == (
            other.ai_bitboard, other.game_bitboard, other.depth % 2)

    # def bitboard_to_array(self, ai_bitboard: int, human_bitboard: int) -> np.ndarray:
    #     output = np.zeros((49,), dtype=np.int8)
    #     for i in range(49):
    #         is_played = (ai_bitboard >> i) & 1
    #         if is_played:
    #             player = (human_bitboard >> i) & 1
    #             output[i] = 1 if player == 0 else 2
    #     return output.reshape((7,7))
    
    # def bitboard_to_array(self, bitboard):
    #     s = 7 * np.arange(6, -1, -1, dtype=np.uint64)
    #     print(s)
    #     b = (bitboard >> s).astype(np.uint8)
    #     print(b)
    #     b = np.unpackbits(b, bitorder="little")
    #     return b.reshape(7, 7)
    
    def make_move(self, position, mask, col):
        """ Helper method to make a move and return new position along with new board position """
        opponent_position = position ^ mask
        new_mask = mask | (mask + (1 << (col * 7)))
        return opponent_position ^ new_mask, new_mask

    def make_move_opponent(self, position, mask, col):
        """ Helper method to only return new board position """
        new_mask = mask | (mask + (1 << (col * 7)))
        return position, new_mask


