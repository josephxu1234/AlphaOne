# AlphaOne

## View my paper! Click on jx6_final_written_report.pdf!
ChessWrapper.py:
- Wrapper class for the Chess.Board class
- Originally meant to use a generalized API as to support more games, but likely would have to tweak this in the future

evaluation.py:
- Contains the functions needed to perform board, move, piece evaluations

lr_eval.pkl:
- pickle file containing the LinearRegression static evaluation model

M2_puzzles.txt:
- List of mate-in-two puzzles to test tactical ability.

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

Parse Puzzles.ipynb:
- parses the M2_puzzles.txt to extract the FEN representations of the puzzle boards, and their solutions.
