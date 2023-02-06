# Copyright The Lightning AI team.
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
import urllib
from typing import Any, Callable, Optional, Sized, Tuple, Union
from urllib.error import HTTPError
from warnings import warn

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib

_DATASETS_PATH = "./data"


class _MNIST(Dataset):
    """Carbon copy of ``tests_pytorch.helpers.datasets.MNIST``.

    We cannot import the tests as they are not distributed with the package.
    See https://github.com/Lightning-AI/lightning/pull/7614#discussion_r671183652 for more context.
    """

    RESOURCES = (
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    )

    TRAIN_FILE_NAME = "training.pt"
    TEST_FILE_NAME = "test.pt"
    cache_folder_name = "complete"

    def __init__(
        self, root: str, train: bool = True, normalize: tuple = (0.1307, 0.3081), download: bool = True, **kwargs: Any
    ) -> None:
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
        return os.path.join(self.root, "MNIST", self.cache_folder_name)

    def _check_exists(self, data_folder: str) -> bool:
        existing = True
        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            existing = existing and os.path.isfile(os.path.join(data_folder, fname))
        return existing

    def prepare_data(self, download: bool = True) -> None:
        if download and not self._check_exists(self.cached_folder_path):
            self._download(self.cached_folder_path)
        if not self._check_exists(self.cached_folder_path):
            raise RuntimeError("Dataset not found.")

    def _download(self, data_folder: str) -> None:
        os.makedirs(data_folder, exist_ok=True)
        for url in self.RESOURCES:
            logging.info(f"Downloading {url}")
            fpath = os.path.join(data_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)

    @staticmethod
    def _try_load(path_data: str, trials: int = 30, delta: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Resolving loading from the same time from multiple concurrent processes."""
        res, exception = None, None
        assert trials, "at least some trial has to be set"
        assert os.path.isfile(path_data), f"missing file: {path_data}"
        for _ in range(trials):
            try:
                res = torch.load(path_data)
            # todo: specify the possible exception
            except Exception as ex:
                exception = ex
                time.sleep(delta * random.random())
            else:
                break
        assert res is not None
        if exception is not None:
            # raise the caught exception
            raise exception
        return res

    @staticmethod
    def normalize_tensor(tensor: Tensor, mean: Union[int, float] = 0.0, std: Union[int, float] = 1.0) -> Tensor:
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        return tensor.sub(mean).div(std)


def MNIST(*args: Any, **kwargs: Any) -> Dataset:
    torchvision_mnist_available = not bool(os.getenv("PL_USE_MOCKED_MNIST", False))
    if torchvision_mnist_available:
        try:
            from torchvision.datasets import MNIST

            MNIST(_DATASETS_PATH, download=True)
        except HTTPError as e:
            print(f"Error {e} downloading `torchvision.datasets.MNIST`")
            torchvision_mnist_available = False
    if not torchvision_mnist_available:
        print("`torchvision.datasets.MNIST` not available. Using our hosted version")
        MNIST = _MNIST
    return MNIST(*args, **kwargs)


class MNISTDataModule(LightningDataModule):
    """Standard MNIST, train, val, test splits and transforms.

    >>> MNISTDataModule()  # doctest: +ELLIPSIS
    <...mnist_datamodule.MNISTDataModule object at ...>
    """

    name = "mnist"

    def __init__(
        self,
        data_dir: str = _DATASETS_PATH,
        val_split: int = 5000,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            seed: starting seed for RNG.
            batch_size: desired batch size.
        """
        super().__init__(*args, **kwargs)
        if num_workers and _IS_WINDOWS:
            # see: https://stackoverflow.com/a/59680818
            warn(
                f"You have requested num_workers={num_workers} on Windows,"
                " but currently recommended is 0, so we set it for you"
            )
            num_workers = 0

        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        """Saves MNIST files to `data_dir`"""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        """Split the train and valid dataset."""
        extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        dataset: Dataset = MNIST(self.data_dir, train=True, download=False, **extra)
        assert isinstance(dataset, Sized)
        train_length = len(dataset)
        self.dataset_train, self.dataset_val = random_split(dataset, [train_length - self.val_split, self.val_split])

    def train_dataloader(self) -> DataLoader:
        """MNIST train set removes a subset to use for validation."""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """MNIST val set uses a subset of the training set for validation."""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """MNIST test set uses the test split."""
        extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        dataset = MNIST(self.data_dir, train=False, download=False, **extra)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    @property
    def default_transforms(self) -> Optional[Callable]:
        if not _TORCHVISION_AVAILABLE:
            return None
        if self.normalize:
            mnist_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5,), std=(0.5,))]
            )
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms
