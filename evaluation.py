import chess
import pandas as pd

# Piece values in centipawns (e.g. 100 centipawns = 1 'point', where Pawns are generally counted as 1 point colloquially)
# Using RofChade's PeSTO tapered eval numbers
# http://www.talkchess.com/forum3/viewtopic.php?f=2&t=68311&start=19
# MG = Middle Game, EG = End Game

# simplest form of material eval
BASE_VALUES = {
    chess.PAWN : 100,
    chess.BISHOP: 350,
    chess.KNIGHT: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 24000
}

# middle game values
MG_VALUES= {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 24000,
}

# end game values
EG_VALUES = {
    chess.PAWN: 94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK: 512,
    chess.QUEEN: 936,
    chess.KING: 24000,
}

# King Zone attack values
KZ_VALUES = {
    chess.PAWN: 0,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 5,
    chess.KING: 0
}

# MG and EG PSQTs for each piece
MG_PAWN = [
    0,   0,   0,   0,   0,   0,  0,   0,
    98, 134,  61,  95,  68, 126, 34, -11,
    -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0]

EG_PAWN = [
    0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0]

MG_KNIGHT = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23]

EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64]

MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21]

EG_BISHOP = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17]

MG_ROOK = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26]

EG_ROOK = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20]

MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50]

EG_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41]

MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14]

EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43]

# Most Valuable Victim, Least Valuable Attacker Heuristic to sort captures
MVV_LVA = [
    [0, 0, 0, 0, 0, 0, 0],       # victim K, attacker K, Q, R, B, N, P, None
    [50, 51, 52, 53, 54, 55, 0], # victim Q, attacker K, Q, R, B, N, P, None
    [40, 41, 42, 43, 44, 45, 0], # victim R, attacker K, Q, R, B, N, P, None
    [30, 31, 32, 33, 34, 35, 0], # victim B, attacker K, Q, R, B, N, P, None
    [20, 21, 22, 23, 24, 25, 0], # victim N, attacker K, Q, R, B, N, P, None
    [10, 11, 12, 13, 14, 15, 0], # victim P, attacker K, Q, R, B, N, P, None
    [0, 0, 0, 0, 0, 0, 0],       # victim None, attacker K, Q, R, B, N, P, None
]

# map pieces to their PSQTs for middlegame
MG_MAP = {
    chess.PAWN: MG_PAWN,
    chess.KNIGHT: MG_KNIGHT,
    chess.BISHOP: MG_BISHOP,
    chess.ROOK: MG_ROOK,
    chess.QUEEN: MG_QUEEN,
    chess.KING: MG_KING
}

# map pieces to their PSQTs for endgame
EG_MAP = {
    chess.PAWN: EG_PAWN,
    chess.KNIGHT: EG_KNIGHT,
    chess.BISHOP: EG_BISHOP,
    chess.ROOK: EG_ROOK,
    chess.QUEEN: EG_QUEEN,
    chess.KING: EG_KING
}

# higher weight means that the loss of that piece is more indicative of getting sent to an endgame
P_PHASE = 1/8
N_PHASE = 1
B_PHASE = 1
R_PHASE = 2
Q_PHASE = 4

# currently: TOTAL_PHASE = 26
# 16 pawns, 4 knights, 4 bishops, 4 rooks, 2 queens total at start
TOTAL_PHASE = P_PHASE*16 + N_PHASE*4 + B_PHASE*4 + R_PHASE*4 + Q_PHASE*2

def get_num_pieces(board):
    # get pawn counts
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    
    # get knight counts
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))

    # get bishop counts
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))

    # get rook counts
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))

    # get queen counts
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    return (
        (wp, P_PHASE), 
        (bp, P_PHASE), 
        (wn, N_PHASE), 
        (bn, N_PHASE), 
        (wb, B_PHASE), 
        (bb, B_PHASE), 
        (wr, R_PHASE), 
        (br, R_PHASE), 
        (wq, Q_PHASE), 
        (bq, Q_PHASE)
    )
    
def get_phase(state):
    """Return the phase value between [0, 1] of a given state, where lower phase values indicate progressing towards the endgame."""
    phase = TOTAL_PHASE
    pieces = get_num_pieces(state)

    for num_pieces, phase_val in pieces:
        # more pieces = lower phase value
        # higher phase value = closer to endgame
        phase -= num_pieces * phase_val

    phase = (phase/TOTAL_PHASE) # scale with highest possible phase value
    return phase

def evaluate_piece(piece, square, phase):
    """Return the tapered evaluation of a given piece, accounting for its material and positional value (from its PSTs)."""
    mg_score = 0
    eg_score = 0
    if piece.color == chess.WHITE:
        mg_score = MG_MAP[piece.piece_type][56 ^ square] + MG_VALUES[piece.piece_type]
        eg_score = EG_MAP[piece.piece_type][56 ^ square] + EG_VALUES[piece.piece_type]
    else:
        mg_score = MG_MAP[piece.piece_type][square] + MG_VALUES[piece.piece_type]
        eg_score = EG_MAP[piece.piece_type][square] + EG_VALUES[piece.piece_type]

    # return weighted average of mg_score, eg_score by phase
    return mg_score * (1 - phase) + eg_score * (phase)

def tapered_eval(board):
    phase = get_phase(board)

    material_counts = {
        chess.WHITE : 0,
        chess.BLACK : 0
    }
    
    # python-chess defines A1 as 0, H8 as 63
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None: # skip trying to evaluate empty squares
            continue
        piece_val = evaluate_piece(piece, square, phase)
        if piece.color == chess.WHITE:
            material_counts[chess.WHITE] += piece_val
        else:
            material_counts[chess.BLACK] += piece_val
    
    return material_counts[chess.WHITE] - material_counts[chess.BLACK]

# def tapered_eval(state):
#     """Return a heuristic estimate of the current position. Higher values favor White, lower values favor Black."""
#     outcome = state.outcome()

#     # in case the game has ended.
#     if outcome is not None:
#         # return large magnitude constant for case of checkmate
#         if outcome.winner == chess.WHITE:
#             return BASE_VALUES[chess.KING]
#         elif outcome.winner == chess.BLACK:
#             return -BASE_VALUES[chess.KING]
#         else: # draw
#             return 0 
#     phase = get_phase(state)

#     material_counts = {
#         chess.WHITE : 0,
#         chess.BLACK : 0
#     }

#     # python-chess defines A1 as 0, H8 as 63
#     for square in range(64):
#         piece = state.piece_at(square)
#         if piece is None: # skip trying to evaluate empty squares
#             continue
#         piece_val = evaluate_piece(piece, square, phase)
#         if piece.color == chess.WHITE:
#             material_counts[chess.WHITE] += piece_val
#         else:
#             material_counts[chess.BLACK] += piece_val

#     return material_counts[chess.WHITE] - material_counts[chess.BLACK]

def mobility(state):
    """Mobility is estimated using the number of legal moves. Returns White's mobility - Black's mobility."""
    
    mobility1 = len(state.get_legal_moves())

    # Change side to move by pushing a null move
    state.push(chess.Move.null())
    
    mobility2 = len(state.get_legal_moves())
    
    # Take back the null move to reset the board back to the position
    state.pop()

    # if turn is white, mobility 1 represents white's mobility
    mobility_delta = mobility1 - mobility2 
    if state.get_turn() == chess.BLACK:
        mobility_delta *= -1
    return mobility_delta

def doubled_pawns(board, color):
    """Return a counter of the number of pairs of pawns occupying the same file for a given color."""
    p_bb = int(board.pieces(chess.PAWN, color)) # get the pawn bitboard
    rows = []
    while p_bb > 0:
        row = p_bb & 0b11111111 # get bottom row       
        p_bb = p_bb >> 8 # remove bottom row
        
        if row > 0:
            rows.append(row) # ignore the rows that are blank
            
    num_doubled = 0
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(rows)):
            compared = rows[i] & rows[j]
            
            if (compared) > 0: # compare each pair of rows
                num_doubled += compared.bit_count() # if two rows have multiple pawns lined up
    return num_doubled

def pawn_islands(board, color):
    """Return the number of pawn islands for a given color."""
    p_bb = int(board.pieces(chess.PAWN, color)) # get the pawn bitboard

    res = 0
    # take bitwise OR of the rows, as if all of the pawns collapse onto 1 rank
    while p_bb > 0:
        row = p_bb & 0b11111111 # get bottom row       
        p_bb = p_bb >> 8 # remove bottom row
        res = res | row
    
    curr_run = 0
    islands = 0
    while res > 0:
        if res & 1 == 1:
            curr_run += 1
        else:
            if curr_run >= 1:
                islands += 1
                curr_run = 0
        res = res >> 1
    if curr_run >= 1:
        islands += 1
    return islands

def passers(board, color):
    """Return the number of passed pawns of a given color."""
    p1_p_bb = int(board.pieces(chess.PAWN, color))
    p2_p_bb = int(board.pieces(chess.PAWN, not color))
    
    p1_rows = []
    p2_rows = []
    for i in range(8):
        row = p1_p_bb & 0b11111111 # get bottom row       
        p1_p_bb = p1_p_bb >> 8 # remove bottom row
        p1_rows.append(row)
        
    for i in range(8):
        row = p2_p_bb & 0b11111111 # get bottom row       
        p2_p_bb = p2_p_bb >> 8 # remove bottom row
        p2_rows.append(row)
        
    if color == chess.BLACK:
        p1_rows.reverse()
        p2_rows.reverse()
        
    passers = 0
    for r in range(1, 7):
        blockers = 0
        if p1_rows[r] > 0:
            for r2 in range(r + 1, 7):
                # for each row ahead of the p1's current row, check if there's a blocker
                blockers = blockers | p2_rows[r2] | (p2_rows[r2] << 1) | (p2_rows[r2] >> 1)
        passers += (~blockers & p1_rows[r]).bit_count()
    
    return passers

def king_safety(board, color):
    """Return the attack value against our king, and our pawn shield value."""
    k = board.king(color) # get king's square
    kr = chess.square_rank(k) # get king's rank
    kf = chess.square_file(k) # get king's file
    
    
    king_zone = board.attacks(k)
    if color == chess.WHITE:
        delta = [(2, -1), (2, 0), (2, 1)]
    else:
        delta = [(-2, -1), (-2, 0), (-2, 1)]
        
    for d in delta:
        r = kr + d[0]
        f = kf + d[1]
        if r >= 0 and r <= 7 and f >= 0 and f <= 7:
            king_zone.add(chess.square(f, r))

    # attack value increases the more pieces we have pointed at the enemy king zone
    atk_value = 0
    # king zone defined as the squares the king can reach + 3 more forward squares facing enemy position
    pawn_shield = 0
    
    for sq in king_zone:
        sq_r = chess.square_rank(sq)

        # check pawn shield squares
        if chess.WHITE:
            if sq_r == kr + 1 and board.piece_type_at(sq) == chess.PAWN:
                pawn_shield += 1
            elif sq_r == kr + 2 and board.piece_type_at(sq) == chess.PAWN:
                pawn_shield += 0.5 # slightly less good, but not terrible
        else:
            if sq_r == kr - 1 and board.piece_type_at(sq) == chess.PAWN:
                pawn_shield += 1
            elif sq_r == kr - 2 and board.piece_type_at(sq) == chess.PAWN:
                pawn_shield += 0.5 # slightly less good, but not terrible
        
        atks = board.attackers(not color, sq)        
        for atk_sq in atks:
            if board.piece_at(atk_sq) is None:
                continue
            atk_value += KZ_VALUES[board.piece_type_at(atk_sq)]
            
    return atk_value, pawn_shield

def evaluate_capture(board: chess.Board, move: chess.Move) -> float:
    """
    Given a capturing move, weight the trade being made.
    """
    if board.is_en_passant(move):
        return BASE_VALUES[chess.PAWN]
    to_piece = board.piece_at(move.to_square)
    from_piece = board.piece_at(move.from_square)
    if to_piece is None or from_piece is None:
        raise Exception('Error, piece not found')
    return BASE_VALUES[to_piece.piece_type] - BASE_VALUES[from_piece.piece_type]

def PST_val(piece, square, endgame):
    if piece.color == chess.WHITE:
        if endgame:
            return EG_MAP[piece.piece_type][56 ^ square]

        return MG_MAP[piece.piece_type][56 ^ square]
    else:
        if endgame:
            return EG_MAP[piece.piece_type][square]
        
        return MG_MAP[piece.piece_type][square]

def mvv_lva_idx(piece_type):
    """Helper function for lookups in the MVV_LVA table."""
    if piece_type == chess.KING:
        return 0
    elif piece_type == chess.QUEEN:
        return 1
    elif piece_type == chess.ROOK:
        return 2
    elif piece_type == chess.BISHOP:
        return 3
    elif piece_type == chess.KNIGHT:
        return 4
    elif piece_type == chess.PAWN:
        return 5
    else:
        return 6

def mvv_lva(board, move):
    """Return the MVV-LVA value of a move. In general, lower value pieces capturing higher value pieces is better."""
    atk_piece = board.piece_at(move.from_square)
    def_piece = board.piece_at(move.to_square)

    if atk_piece is None or def_piece is None:
        return 0
        
    atk_idx = mvv_lva_idx(atk_piece.piece_type)
    def_idx = mvv_lva_idx(def_piece.piece_type)
    return MVV_LVA[def_idx][atk_idx]

def move_value(board, move):
    """
    Return a value for a given move on the board.
    Incentivize lower pieces capturing higher pieces, checks, and good positional changes.
    """
    if move.promotion is not None:
        return float("inf")

    piece = board.piece_at(move.from_square)

    # endgame if neither side has a queen
    endgame = True if (board.pieces(chess.QUEEN, chess.BLACK) == 0 and board.pieces(chess.QUEEN, chess.WHITE) == 0) else False
        
    if piece:
        from_value = PST_val(piece, move.from_square, endgame)
        to_value = PST_val(piece, move.to_square, endgame)
        position_change = to_value - from_value
    else:
        raise Exception('Piece not found')

    capture_value = 0.0
    if board.is_capture(move):
        capture_value = evaluate_capture(board, move)

    check_value = 0.0
    if board.gives_check(move):
        check_value = 50.0

    current_move_value = capture_value + position_change + check_value

    return current_move_value

def get_ordered_moves(board, moves):
    """
    Get legal moves, and try to sort them from best to worse.
    Use piece values, PST changes to weight moves.
    """
    def sorter(move):
        return move_value(board, move)

    in_order = sorted(moves, key=sorter, reverse=True)
    return list(in_order)

def get_ordered_captures(board, moves):
    """
    Get legal moves, and try to sort them from best to worse.
    Use MVV-LVA table.
    """
    def sorter(move):
        return mvv_lva(board, move)

    in_order = sorted(moves, key=sorter, reverse=True)
    return list(in_order)

def create_features(board):
    """
    Return features of the board for the regression model to use
    """
    
    b_atk, w_ps = king_safety(board, chess.WHITE)
    w_atk, b_ps = king_safety(board, chess.BLACK)

    new_row = pd.DataFrame(
        {
            'tapered_eval': [tapered_eval(board)],
            'king_atk': [w_atk - b_atk],
            'mobility' : [mobility(board)],
            'pawn_shield': [w_ps - b_ps],
            'pawn_islands' : [pawn_islands(board, chess.WHITE) - pawn_islands(board, chess.BLACK)],
            'doubled_pawns' : [doubled_pawns(board, chess.WHITE) - doubled_pawns(board, chess.BLACK)],
            'passed_pawns' : [passers(board, chess.WHITE) - passers(board, chess.BLACK)],

        }
    )

    return new_row
