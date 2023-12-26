import chess.pgn
import json
import torch
from torch.utils.data import Dataset, DataLoader


def parse_pgn_to_moves(pgn_file, maxlen=False):
    games = []
    seqlen = 0
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            # getting Universal Chess Interface for each move in the game
            moves = [move.uci() for move in game.mainline_moves()]
            if maxlen:
                seqlen = max(seqlen, len(moves))
            if len(moves):
                games.append(moves)
    if maxlen:
        return games, seqlen
    return games


def parse_moves_lists(list_file, maxlen=False, num_games=None):
    with open(list_file, 'r') as file:
        games = file.readlines()
    if not num_games is None and num_games < len(games):
        games = games[:num_games]
    games = [moves.split() for moves in games]
    games = [game for game in games if len(game)]
    mlen = 0
    if maxlen:
        for game in games:
            mlen = max(len(game), mlen)
        return games, mlen
    return games


# You might need a function to tokenize the moves
def tokenize_moves(moves, token_to_id):
    return [token_to_id[move] for move in moves if move in token_to_id]


class ChessDataset(Dataset):


    def __init__(self, tokenized_games, players, pieces, start_pos, end_pos, sequence_length):
        # Assuming tokenized_games is already tokenized and padded as required
        self.games = [self.add_padding(game, sequence_length) for game in tokenized_games]
        self.players = [self.add_padding(game, sequence_length) for game in players]
        self.pieces = [self.add_padding(game, sequence_length) for game in pieces]
        self.start_pos = [self.add_padding(game, sequence_length) for game in start_pos]
        self.end_pos = [self.add_padding(game, sequence_length) for game in end_pos]
        self.sequence_length = sequence_length

    @staticmethod
    def add_padding(game, sequence_length):
        # Efficient padding with PyTorch operations
        padded_game = torch.full((sequence_length,), fill_value=0, dtype=torch.long)
        length = min(len(game), sequence_length)
        padded_game[:length] = torch.tensor(game[:length], dtype=torch.long)
        return padded_game

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        players = self.players[idx]
        pieces = self.pieces[idx]
        start_pos = self.start_pos[idx]
        end_pos = self.end_pos[idx]

        x = game[:-1]  # Input sequence
        players = players[:-1]
        pieces = pieces[:-1]
        start_pos = start_pos[:-1]
        end_pos = end_pos[:-1]
        y = game[1:]   # Target sequence
        return x, players, pieces, start_pos, end_pos, y


piece_to_index = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}
square_to_index = {'a1': 1, 'b1': 2, 'c1': 3, 'd1': 4, 'e1': 5, 'f1': 6, 'g1': 7, 'h1': 8, 'a2': 9, 'b2': 10, 'c2': 11, 'd2': 12, 'e2': 13, 'f2': 14, 'g2': 15, 'h2': 16, 'a3': 17, 'b3': 18, 'c3': 19, 'd3': 20, 'e3': 21, 'f3': 22, 'g3': 23, 'h3': 24, 'a4': 25, 'b4': 26, 'c4': 27, 'd4': 28, 'e4': 29, 'f4': 30, 'g4': 31, 'h4': 32, 'a5': 33, 'b5': 34, 'c5': 35, 'd5': 36, 'e5': 37, 'f5': 38, 'g5': 39, 'h5': 40, 'a6': 41, 'b6': 42, 'c6': 43, 'd6': 44, 'e6': 45, 'f6': 46, 'g6': 47, 'h6': 48, 'a7': 49, 'b7': 50, 'c7': 51, 'd7': 52, 'e7': 53, 'f7': 54, 'g7': 55, 'h7': 56, 'a8': 57, 'b8': 58, 'c8': 59, 'd8': 60, 'e8': 61, 'f8': 62, 'g8': 63, 'h8': 64}


def get_players(line):
    def get_player(mv):
        return 1 + int(mv[0] == mv[0].lower())
    return [get_player(mv) for mv in line]


def get_pieces(line):
    return [piece_to_index[mv[0].lower()] for mv in line]


def get_pos(line, start):
    return [square_to_index[mv[start:start+2]] for mv in line]

def get_embedded_lists(games):
    players = [get_players(game) for game in games]
    pieces = [get_pieces(game) for game in games]
    start_pos = [get_pos(game, 1) for game in games]
    end_pos = [get_pos(game, 3) for game in games]
    return players, pieces, start_pos, end_pos

def get_dataloader(pgn_file, token_id_file, batch_size=32, sequence_len=0, num_workers=4, num_games=None):
    if sequence_len:
        games = parse_moves_lists(pgn_file, maxlen=False, num_games=num_games)
    else:
        games, sequence_len = parse_moves_lists(pgn_file, maxlen=True)

    with open(token_id_file, 'r') as file:
        token_to_id = json.load(file)  # Define your token to id mapping

    # Tokenize moves
    tokenized_games = [tokenize_moves(game, token_to_id) for game in games]
    players, pieces, start_pos, end_pos = get_embedded_lists(games)

    dataset = ChessDataset(tokenized_games, players, pieces, start_pos, end_pos, sequence_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)  # try up to 8
    return dataloader


if __name__ == '__main__':
    print(get_dataloader('path_to_your_pgn_file.pgn'))
