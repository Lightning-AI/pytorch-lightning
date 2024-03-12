from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int) -> None:
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int) -> None:
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(self.count):
            yield torch.randn(self.size)
