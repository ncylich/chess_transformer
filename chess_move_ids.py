import chess.pgn
import json

import time
from datetime import timedelta

from copy import deepcopy

SEARCH = 0

OUTFILE = "chess_move_ids.json"
INFILE = "lichess_db_standard_rated_2023-11.pgn"

# UCI PROTOCL
# start_square + end_square
    # if promotionm, suffix of q,r,b, or n

start_time = time.time()

unique_moves = { '0000' }

if SEARCH:
    rng = 10**6

    with open(INFILE) as f:
        for i in range(rng):
            game = chess.pgn.read_game(f)
            if i % (rng/100) == 0:
                elapsed = str(timedelta(seconds=time.time() - start_time))
                print(f"{100*i/rng}% elapsed time: ", elapsed)
            if game is None:
                break
            # getting Universal Chess Interface for each move in the game
            moves = [move.uci() for move in game.mainline_moves()]
            for move in moves:
                unique_moves.add(move)


else:
    unique_spots = set()
    for i in range(8):
        for j in range(1, 9):
            spot = chr(ord('a') + i) + str(j)
            unique_spots.add(spot)

    for start in unique_spots:
        for end in unique_spots:
            dx = abs(ord(start[0]) - ord(end[0]))
            dy = abs(int(start[1]) - int(end[1]))
            # must actually move
            if dx == 0 and dy == 0:
                continue
            # must move up or down or diagonally or like a knight
            if dx == 0 or dy == 0 or dx == dy or dx + dy == 3:
                move = start + end
                unique_moves.add(move)

    final_moves = deepcopy(unique_moves)
    for move in unique_moves:
        start, end = move[:2], move[2:]
        if (start[1] == '2' and end[1] == '1') or (start[1] == '7' and end[1] == '8'):
            if abs(ord(start[0])-ord(end[0])) <= 1:
                final_moves.add(f'{move}q')   # queen
                final_moves.add(f'{move}r')  # rook
                final_moves.add(f'{move}b')  # bishop
                final_moves.add(f'{move}n')  # knight
    unique_moves = final_moves

token_to_id = {move: i+1 for i, move in enumerate(unique_moves)}
print('# of unique moves:', len(unique_moves))

if 1:
    with open(OUTFILE, "w") as file:
        json.dump(token_to_id, file)

elapsed = str(timedelta(seconds=time.time() - start_time))
print("Elapsed Time: ", elapsed)
