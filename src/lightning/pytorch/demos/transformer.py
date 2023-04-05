"""Demo of a simple transformer language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model
"""
import math
import os
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class WikiText2(Dataset):
    """Mini version of WikiText2."""

    def __init__(self, data_dir=Path("./data"), block_size=35, download=True):
        super().__init__()
        self.path = data_dir / "wikitext-2.txt"
        if download:
            self.download(self.path)
        self.data, self.dictionary = tokenize(self.path)
        self.block_size = block_size

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)

    def __len__(self) -> int:
        return len(self.data) // self.block_size - 1

    def __getitem__(self, index):
        start = index * self.block_size
        end = start + self.block_size
        input = self.data[start:end]
        target = self.data[(start + 1) : (end + 1)]
        return input, target

    @staticmethod
    def download(destination: Path) -> None:
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        if os.path.exists(destination):
            return
        with open(destination, "w") as f:
            f.write(requests.get(url).text)


class Transformer(nn.Module):
    def __init__(self, ntokens, ninp=200, nhead=2, nhid=200, nlayers=2, dropout=0.2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, ninp)
        self.decoder = nn.Linear(ninp, ntokens)

        self.ninp = ninp
        self.ntokens = ntokens
        self.src_mask = None
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask and (self.src_mask is None or self.src_mask.size(0) != len(src)):
            mask = (torch.triu(torch.ones(len(src), len(src), device=src.device)) == 1).transpose(0, 1)
            self.src_mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        output = output.view(-1, self.ntokens)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def tokenize(path):
    dictionary = Dictionary()

    assert os.path.exists(path)
    # Add words to the dictionary
    with open(path, encoding="utf8") as f:
        for line in f:
            words = line.split() + ["<eos>"]
            for word in words:
                dictionary.add_word(word)

    # Tokenize file content
    with open(path, encoding="utf8") as f:
        idss = []
        for line in f:
            words = line.split() + ["<eos>"]
            ids = []
            for word in words:
                ids.append(dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

    return ids, dictionary
