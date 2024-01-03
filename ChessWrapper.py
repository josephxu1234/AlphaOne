from chess import Board
import chess
from material_values import *
import numpy as np

class ChessWrapper(Board):
    """
    Wrapper Class for Chess Boards, to support the following functions:

    is_game_over(): return if the game has completed
    push(move): pushes a move onto the current board
    outcome(): return the outcome of the game
    get_legal_moves(): return a list of legal moves from the current position
    get_turn(): return the current side to move
    """
    # 
    def get_legal_moves(self):
        """Return a list of the legal moves from the current position"""
        return list(self.legal_moves)

    def get_turn(self):
        """Return whose turn to play. chess.WHITE = True, chess.BLACK = false."""
        return self.turn            