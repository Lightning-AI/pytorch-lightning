"""Demo of a simple transformer language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model
"""
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


class Transformer(nn.Module):
    def __init__(
        self, vocab_size: int, ninp: int = 200, nhead: int = 2, nhid: int = 200, nlayers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.embedding = nn.Embedding(vocab_size, ninp)
        self.transformer = nn.Transformer(
            d_model=ninp,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(ninp, vocab_size)

        self.ninp = ninp
        self.vocab_size = vocab_size
        self.src_mask = None

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        b, t = input.shape

        # we assume target is already shifted w.r.t. input
        if mask is None:
            mask = torch.tril(torch.ones(t, t, device=input.device)) == 1
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

        src = self.pos_encoder(self.embedding(input) * math.sqrt(self.ninp))
        target = self.pos_encoder(self.embedding(target) * math.sqrt(self.ninp))
        output = self.transformer(src, target, tgt_mask=mask)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        output = output.view(-1, self.vocab_size)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x + self.pe[: x.size(0), :]  # type: ignore[index]
        return self.dropout(x)


class WikiText2(Dataset):
    """Mini version of WikiText2."""

    def __init__(self, data_dir: Path = Path("./data"), block_size: int = 35, download: bool = True) -> None:
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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        input = self.data[start:end]
        target = self.data[(start + 1) : (end + 1)]
        return input, target

    @staticmethod
    def download(destination: Path) -> None:
        os.makedirs(destination.parent, exist_ok=True)
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        if os.path.exists(destination):
            return
        with open(destination, "w") as f:
            f.write(requests.get(url).text)


class Dictionary:
    def __init__(self) -> None:
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


def tokenize(path: Path) -> Tuple[Tensor, Dictionary]:
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
        idss: List[Tensor] = []
        for line in f:
            words = line.split() + ["<eos>"]
            ids: List[int] = []
            for word in words:
                ids.append(dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))

    return torch.cat(idss), dictionary
