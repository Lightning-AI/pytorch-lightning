# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch.utils.data import Dataset, IterableDataset

from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel, ManualOptimBoringModel, RandomDataset

__all__ = ["BoringDataModule", "BoringModel", "ManualOptimBoringModel", "RandomDataset"]


class RandomDictDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

    def __len__(self):
        return self.len


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self):
        return self.count
