"""Demo of a simple transformer language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model

"""

import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn.modules import MultiheadAttention
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule

_REQUESTS_AVAILABLE = RequirementCache("requests")


if hasattr(MultiheadAttention, "_reset_parameters") and not hasattr(MultiheadAttention, "reset_parameters"):
    # See https://github.com/pytorch/pytorch/issues/107909
    MultiheadAttention.reset_parameters = MultiheadAttention._reset_parameters


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 33278,  # default for WikiText2
        ninp: int = 200,
        nhead: int = 2,
        nhid: int = 200,
        nlayers: int = 2,
        dropout: float = 0.2,
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

    def forward(self, inputs: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        _, t = inputs.shape

        # we assume target is already shifted w.r.t. inputs
        if mask is None:
            mask = torch.tril(torch.ones(t, t, device=inputs.device)) == 1
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

        src = self.pos_encoder(self.embedding(inputs) * math.sqrt(self.ninp))
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
        self.dim = dim
        self.max_len = max_len
        self.pe: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        if self.pe is None:
            # 1) can't use buffer, see https://github.com/pytorch/pytorch/issues/68407
            # 2) can't use parameter becauses pe gets sliced and DDP requires all params to participate in forward
            # TODO: Could make this a `nn.Parameter` with `requires_grad=False`
            self.pe = self._init_pos_encoding(device=x.device)

        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)

    def _init_pos_encoding(self, device: torch.device) -> Tensor:
        pe = torch.zeros(self.max_len, self.dim, device=device)
        position = torch.arange(0, self.max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


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

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        inputs = self.data[start:end]
        target = self.data[(start + 1) : (end + 1)]
        return inputs, target

    @staticmethod
    def download(destination: Path) -> None:
        if not _REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(str(_REQUESTS_AVAILABLE))

        import requests

        os.makedirs(destination.parent, exist_ok=True)
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        if os.path.exists(destination):
            return
        with open(destination, "w") as f:
            f.write(requests.get(url).text)


class Dictionary:
    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


def tokenize(path: Path) -> tuple[Tensor, Dictionary]:
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
        idss: list[Tensor] = []
        for line in f:
            words = line.split() + ["<eos>"]
            ids: list[int] = []
            for word in words:
                ids.append(dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))

    return torch.cat(idss), dictionary


class LightningTransformer(LightningModule):
    def __init__(self, vocab_size: int = 33278) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return self.model(inputs, target)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def prepare_data(self) -> None:
        WikiText2(download=True)

    def train_dataloader(self) -> DataLoader:
        dataset = WikiText2()
        return DataLoader(dataset)
