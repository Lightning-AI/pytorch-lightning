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
import os
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset

from pytorch_lightning.utilities.debug_examples import MNIST


class TrialMNIST(MNIST):
    """Constrained MNIST dataset

    Args:
        num_samples: number of examples per selected class/digit
        digits: list selected MNIST digits/classes
        kwargs: Same as MNIST

    Examples:
        >>> dataset = TrialMNIST(".", download=True)
        >>> len(dataset)
        300
        >>> sorted(set([d.item() for d in dataset.targets]))
        [0, 1, 2]
        >>> torch.bincount(dataset.targets)
        tensor([100, 100, 100])
    """

    def __init__(self, root: str, num_samples: int = 100, digits: Optional[Sequence] = (0, 1, 2), **kwargs):
        # number of examples per class
        self.num_samples = num_samples
        # take just a subset of MNIST dataset
        self.digits = sorted(digits) if digits else list(range(10))

        self.cache_folder_name = f"digits-{'-'.join(str(d) for d in self.digits)}_nb-{self.num_samples}"

        super().__init__(root, normalize=(0.5, 1.0), **kwargs)

    @staticmethod
    def _prepare_subset(full_data: torch.Tensor, full_targets: torch.Tensor, num_samples: int, digits: Sequence):
        classes = {d: 0 for d in digits}
        indexes = []
        for idx, target in enumerate(full_targets):
            label = target.item()
            if classes.get(label, float("inf")) >= num_samples:
                continue
            indexes.append(idx)
            classes[label] += 1
            if all(classes[k] >= num_samples for k in classes):
                break
        data = full_data[indexes]
        targets = full_targets[indexes]
        return data, targets

    def _download(self, data_folder: str) -> None:
        super()._download(data_folder)
        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            path_fname = os.path.join(self.cached_folder_path, fname)
            assert os.path.isfile(path_fname), f"Missing cached file: {path_fname}"
            data, targets = self._try_load(path_fname)
            data, targets = self._prepare_subset(data, targets, self.num_samples, self.digits)
            torch.save((data, targets), os.path.join(self.cached_folder_path, fname))


class AverageDataset(Dataset):
    def __init__(self, dataset_len=300, sequence_len=100):
        self.dataset_len = dataset_len
        self.sequence_len = sequence_len
        self.input_seq = torch.randn(dataset_len, sequence_len, 10)
        top, bottom = self.input_seq.chunk(2, -1)
        self.output_seq = top + bottom.roll(shifts=1, dims=-1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item], self.output_seq[item]


class SklearnDataset(Dataset):
    def __init__(self, x, y, x_type, y_type):
        self.x = x
        self.y = y
        self._x_type = x_type
        self._y_type = y_type

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=self._x_type), torch.tensor(self.y[idx], dtype=self._y_type)

    def __len__(self):
        return len(self.y)


class RandomDictDataset(Dataset):
    def __init__(self, size, length):
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
