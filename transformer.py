import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, LayerNorm, Dropout, Linear
from torch.utils.data import dataset


# standard decoder, but with cross-attention layer removed
class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, nplayers=2, npieces=6, npositions=64, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = CustomDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)

        self.player_embedding = nn.Embedding(nplayers + 1, d_model)
        self.piece_embedding = nn.Embedding(npieces + 1, d_model)
        self.position_embedding = nn.Embedding(npositions + 1, d_model)

        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = .1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def org_forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)

        # Since this is a decoder-only model, we don't use a separate memory tensor
        output = self.transformer_decoder(src, memory=None, tgt_mask=src_mask)
        output = self.linear(output)
        return output

    def forward(self, src: Tensor, players: Tensor, pieces: Tensor, start_pos: Tensor, end_pos: Tensor,
                src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src)
        player_emb = self.player_embedding(players)
        piece_emb = self.piece_embedding(pieces)
        start_pos_emb = self.position_embedding(start_pos)
        end_pos_emb = self.position_embedding(end_pos)

        # Combining embeddings
        combined_emb = src + player_emb + piece_emb + start_pos_emb + end_pos_emb

        # Rest of the forward method
        combined_emb = combined_emb * math.sqrt(self.d_model)
        combined_emb = self.pos_encoder(combined_emb)
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(len(combined_emb)).to(combined_emb.device)

        output = self.transformer_decoder(combined_emb, memory=None, tgt_mask=src_mask)
        output = self.linear(output)
        return output

    @staticmethod # GPT4 version
    def _generate_square_subsequent_mask(size: int) -> Tensor:
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
