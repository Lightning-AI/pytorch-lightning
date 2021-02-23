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
import logging
import os
import random
import time
import urllib.request
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tests import _PROJECT_ROOT

#: local path to test datasets
PATH_DATASETS = os.path.join(_PROJECT_ROOT, 'Datasets')


class MNIST(Dataset):
    """
    Customized `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.

    Part of the code was copied from
    https://github.com/pytorch/vision/blob/build/v0.5.0/torchvision/datasets/mnist.py

    Args:
        root: Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    Examples:
        >>> dataset = MNIST(download=True)
        >>> len(dataset)
        60000
        >>> torch.bincount(dataset.targets)
        tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
    """

    RESOURCES = (
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    )

    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    cache_folder_name = 'complete'

    def __init__(
        self,
        root: str = PATH_DATASETS,
        train: bool = True,
        normalize: tuple = (0.1307, 0.3081),
        download: bool = True,
    ):
        super().__init__()
        self.root = root
        self.train = train  # training set or test set
        self.normalize = normalize

        self.prepare_data(download)

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = self._try_load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].float().unsqueeze(0)
        target = int(self.targets[idx])

        if self.normalize is not None and len(self.normalize) == 2:
            img = self.normalize_tensor(img, *self.normalize)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.root, 'MNIST', self.cache_folder_name)

    def _check_exists(self, data_folder: str) -> bool:
        existing = True
        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            existing = existing and os.path.isfile(os.path.join(data_folder, fname))
        return existing

    def prepare_data(self, download: bool = True):
        if download and not self._check_exists(self.cached_folder_path):
            self._download(self.cached_folder_path)
        if not self._check_exists(self.cached_folder_path):
            raise RuntimeError('Dataset not found.')

    def _download(self, data_folder: str) -> None:
        os.makedirs(data_folder)
        for url in self.RESOURCES:
            logging.info(f'Downloading {url}')
            fpath = os.path.join(data_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)

    @staticmethod
    def _try_load(path_data, trials: int = 30, delta: float = 1.):
        """Resolving loading from the same time from multiple concurrent processes."""
        res, exception = None, None
        assert trials, "at least some trial has to be set"
        assert os.path.isfile(path_data), f'missing file: {path_data}'
        for _ in range(trials):
            try:
                res = torch.load(path_data)
            # todo: specify the possible exception
            except Exception as e:
                exception = e
                time.sleep(delta * random.random())
            else:
                break
        if exception is not None:
            # raise the caught exception
            raise exception
        return res

    @staticmethod
    def normalize_tensor(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        return tensor.sub(mean).div(std)


class TrialMNIST(MNIST):
    """Constrained MNIST dataset

    Args:
        num_samples: number of examples per selected class/digit
        digits: list selected MNIST digits/classes
        kwargs: Same as MNIST

    Examples:
        >>> dataset = TrialMNIST(download=True)
        >>> len(dataset)
        300
        >>> sorted(set([d.item() for d in dataset.targets]))
        [0, 1, 2]
        >>> torch.bincount(dataset.targets)
        tensor([100, 100, 100])
    """

    def __init__(self, num_samples: int = 100, digits: Optional[Sequence] = (0, 1, 2), **kwargs):
        # number of examples per class
        self.num_samples = num_samples
        # take just a subset of MNIST dataset
        self.digits = sorted(digits) if digits else list(range(10))

        self.cache_folder_name = f"digits-{'-'.join(str(d) for d in self.digits)}_nb-{self.num_samples}"

        super().__init__(normalize=(0.5, 1.0), **kwargs)

    @staticmethod
    def _prepare_subset(full_data: torch.Tensor, full_targets: torch.Tensor, num_samples: int, digits: Sequence):
        classes = {d: 0 for d in digits}
        indexes = []
        for idx, target in enumerate(full_targets):
            label = target.item()
            if classes.get(label, float('inf')) >= num_samples:
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
            assert os.path.isfile(path_fname), f'Missing cached file: {path_fname}'
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
