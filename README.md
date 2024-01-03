# AlphaOne

ChessWrapper.py:
- Wrapper class for the Chess.Board class
- Originally meant to use a generalized API as to support more games, but likely would have to tweak this in the future

Model Development.ipynb:
- Uses data from https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
- Tunes feature weights using LinearRegression model from sklearn

AlphaBetaEngine.ipynb:
- Implements AlphaBeta with: quiescence, move ordering
- Tests AlphaBeta's ELO with tapered evaluation, custom static evaluation built with regression

AlphaOne MCTS Engine.ipynb:
- Implements a Truncated version of MCTS, emulating AlphaZero's approach
- Key difference: uses static evaluation in place of value network
- Tests AlphaOne's ELO with tapered evaluation, custom static evaluation built with regression
