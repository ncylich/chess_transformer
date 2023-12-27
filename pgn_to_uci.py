import chess.pgn
import time
import os
from datetime import timedelta

LABEL = True
FILTER_NUM = 1800
FILTER = False

INFILE = f"elo_{FILTER_NUM}.pgn"
OUTFILE = f"labeled_elo_{FILTER_NUM}.txt"

start_time = time.time()


def get_piece_symbol(piece, idx):
    """ Return the symbol for the piece. """
    if piece == chess.PAWN:
        ret = 'p'
    elif piece == chess.KNIGHT:
        ret = 'n'
    elif piece == chess.BISHOP:
        ret = 'b'
    elif piece == chess.ROOK:
        ret = 'r'
    elif piece == chess.QUEEN:
        ret = 'q'
    elif piece == chess.KING:
        ret = 'k'
    else:
        ret = 'unknown'
    if idx % 2:
        return ret
    return ret.upper()


def prefix_moves_with_pieces(uci_moves):
    board = chess.Board()
    prefixed_moves = []

    for idx, move in enumerate(uci_moves):
        move_obj = chess.Move.from_uci(move)
        if move_obj in board.legal_moves:
            piece_type = board.piece_type_at(move_obj.from_square)
            piece_symbol = get_piece_symbol(piece_type, idx)
            prefixed_move = f"{piece_symbol}{move}"
            prefixed_moves.append(prefixed_move)
            board.push(move_obj)
        else:
            # If the move is not legal, add it as is.
            prefixed_moves.append(move)
            raise Exception('Illegal Move')

    return prefixed_moves


if FILTER:
    print('Filtering Games First...')
    new_infile = "filtered_" + INFILE
    cmd = f"./filter_file {FILTER_NUM} {INFILE} {new_infile}"
    result = os.system(cmd)
    INFILE = new_infile
    elapsed = str(timedelta(seconds=time.time() - start_time))
    print("Elapsed Time to Filter Games: ", elapsed, '\n')

with open(OUTFILE, 'w') as file:
    with open(INFILE) as pgn:
        count = 1
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            moves = [move.uci() for move in game.mainline_moves()]
            if len(moves) == 0:
                continue

            if LABEL:
                moves = prefix_moves_with_pieces(moves)

            file.write(' '.join(moves) + '\n')
            count += 1

            if count % 1000 == 0:
                print(f'num {count}')
                elapsed = str(timedelta(seconds=time.time() - start_time))
                print("Elapsed Time: ", elapsed)
