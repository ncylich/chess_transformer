from chess_loader import get_embedded_lists
from transformer import TransformerModel
from copy import deepcopy
import torch
import json
import chess

MODEL = "engine_1.2_embedded_whole.pth"
TOKEN_IDS = "all_chess_move_ids.json"

ASK_FOR_MOVE = False
PRINT_BOARD = True


# loading model
model = torch.load(f'models/{MODEL}')
device = next(model.parameters()).device  # Get the device model is on

# loading id-token translation
with open(TOKEN_IDS, 'r') as file:
    token_to_id = json.load(file)
id_to_token = {id_num: token for token, id_num in token_to_id.items()}


# Sample chess game data (list of moves)
input_data = ["Pe2e4", "pe7e5", "Ng1f3"]  # Example moves


def preprocess_data(chess_moves, token_to_id, sequence_length, device):
    tokenized_moves = [token_to_id[move] for move in chess_moves]
    tokenized_moves += [0] * (sequence_length - len(tokenized_moves))
    mv_tensor = torch.tensor(tokenized_moves, dtype=torch.long).unsqueeze(0)
    mv_tensor = mv_tensor.to(device)

    embedded_lists = list(get_embedded_lists([chess_moves]))
    for i, arr in enumerate(embedded_lists):
        arr = arr[0]
        arr += [0] * (sequence_length - len(arr))
        arr = torch.tensor(arr, dtype=torch.long).unsqueeze(0)
        embedded_lists[i] = arr.to(device)

    return mv_tensor, *embedded_lists


def valid_max(output_line, id_to_token, board, line_num, input_data, print_board=False):
    mx_val = None
    mx_move = None

    if ASK_FOR_MOVE:
        input()

    if line_num < len(input_data) - 1:
        mx_move = input_data[line_num + 1]
    else:
        for idx, item in enumerate(output_line):
            if mx_val is None or item > mx_val:
                if idx not in id_to_token:
                    continue
                uci_move = id_to_token[idx]
                move = chess.Move.from_uci(uci_move[1:])

                if board.is_legal(move):
                    mx_val = item
                    mx_move = uci_move

    if not board.is_checkmate():
        board.push(chess.Move.from_uci(mx_move[1:]))
        if print_board:
            print(board, '\n')
    return mx_move


def postprocess_output(output, id_to_token, board, print_board=False):
    # Assuming output is of shape [batch_size, seq_len, n_classes], and batch_size is 1
    # We first select the output of the first item in the batch
    # Convert the single output tensor to a list of token IDs
    # single_output = output[0]
    # token_ids = single_output.argmax(dim=-1).tolist()

    single_output = output[0].cpu().numpy()
    if print_board:
        print(board, '\n')

    # playing first move (engine won't predict this one, might want to implement start token)
    if input_data:
        board.push(chess.Move.from_uci(input_data[0][1:]))
        if print_board:
            print(board, '\n')

    # gets maximum valid move for each token in predicted output
    custom_max = [valid_max(line, id_to_token, board, line_num, input_data, print_board=print_board)
                  for line_num, line in enumerate(single_output)]
    return custom_max

def process_next_token(idx, wht_turn, output, id_to_token, board):
    single_output = output[0].cpu().numpy()
    move_output = single_output[idx - 1]

    mx_val = None
    mx_move = None

    for idx, item in enumerate(move_output):
        if mx_val is None or item > mx_val:
            if idx not in id_to_token:
                continue

            uci_move = id_to_token[idx]
            if (uci_move[0] == uci_move[0].upper()) != wht_turn:
                continue

            move = chess.Move.from_uci(uci_move[1:])

            if board.is_legal(move):
                mx_val = item
                mx_move = uci_move, idx

    return mx_move

def gen_moves(input_data, input_tensors, id_to_token, board, print_board=False):
    for move in input_data:
        board.push(chess.Move.from_uci(move[1:]))
    if print_board:
        print(board, '\n')
    length = len(input_data)
    while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
        if ASK_FOR_MOVE:
            input()

        with torch.no_grad():
            output = model(*input_tensors)

        wht_turn = input_data[-1][0]
        wht_turn = wht_turn == wht_turn.lower()  # seeing if last move was black
        next_move, next_id = process_next_token(length, wht_turn, output, id_to_token, board)

        input_data.append(next_move)
        length += 1

        input_tensors = preprocess_data(input_data, token_to_id, length + 1, device)

        board.push(chess.Move.from_uci(next_move[1:]))
        if print_board:
            print(length)
            print(board, '\n')

    if print_board:
        print(board.outcome())

    return input_data


# Preprocess the datas
input_tensors = preprocess_data(input_data, token_to_id, 150, device)

board = chess.Board()
generated_moves = gen_moves(input_data, input_tensors, id_to_token, board, print_board=PRINT_BOARD)
print("All Chess Moves:", generated_moves)

'''
# Perform inference
with torch.no_grad():
    output = model(*input_tensors)

# Postprocess and display the result
board = chess.Board()
chess_moves = postprocess_output(output, id_to_token, board, print_board=True)
'''

