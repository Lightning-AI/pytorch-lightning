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
"""
Debugging helpers meant to be used ONLY in the ``pl_examples`` and ``tests`` directories.

Production usage is discouraged. No backwards-compatibility guarantees.
"""
import logging
import os
import random
import time
import urllib.request
from typing import Optional, Tuple
from urllib.error import HTTPError

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        """
        Testing PL Module

        Use as follows:
        - subclass
        - modify the behavior for what you want

        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing

        or:

        model = BaseTestModel()
        model.training_epoch_end = None

        """
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x["x"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class BoringDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None
        self.checkpoint_state: Optional[str] = None

    def prepare_data(self):
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.random_train = Subset(self.random_full, indices=range(64))
            self.dims = self.random_train[0].shape

        if stage in ("fit", "validate") or stage is None:
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))

        if stage == "test" or stage is None:
            self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))
            self.dims = getattr(self, "dims", self.random_test[0].shape)

        if stage == "predict" or stage is None:
            self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))
            self.dims = getattr(self, "dims", self.random_predict[0].shape)

    def train_dataloader(self):
        return DataLoader(self.random_train)

    def val_dataloader(self):
        return DataLoader(self.random_val)

    def test_dataloader(self):
        return DataLoader(self.random_test)

    def predict_dataloader(self):
        return DataLoader(self.random_predict)


class MNIST(torch.utils.data.Dataset):
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
        >>> dataset = MNIST(".", download=True)
        >>> len(dataset)
        60000
        >>> torch.bincount(dataset.targets)
        tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
    """

    RESOURCES = (
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    )

    TRAIN_FILE_NAME = "training.pt"
    TEST_FILE_NAME = "test.pt"
    cache_folder_name = "complete"

    def __init__(
        self, root: str, train: bool = True, normalize: tuple = (0.1307, 0.3081), download: bool = True, **kwargs
    ):
        super().__init__()

        self.root = root
        self.train = train  # training set or test set
        self.normalize = normalize

        _USE_MOCKED_MNIST = bool(os.getenv("PL_USE_MOCKED_MNIST", False))
        if not _USE_MOCKED_MNIST:
            # avoid users accidentally importing this MNIST implementation
            self._torchvision_available(self.cached_folder_path)
        self.prepare_data(download)

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = self._try_load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
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

    def prepare_data(self, download: bool = True):
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
    def _try_load(path_data, trials: int = 30, delta: float = 1.0):
        """Resolving loading from the same time from multiple concurrent processes."""
        res, exception = None, None
        assert trials, "at least some trial has to be set"
        assert os.path.isfile(path_data), f"missing file: {path_data}"
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
    def normalize_tensor(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        return tensor.sub(mean).div(std)

    @staticmethod
    def _torchvision_available(root: str, should_raise: bool = True) -> bool:
        # local import to facilitate mocking
        from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

        try:
            from torchvision.datasets import MNIST

            MNIST(root, download=True)
            download_successful = True
        except HTTPError as e:
            # allow using our implementation if torchvision hosting is down
            print(f"Error {e} downloading `torchvision.datasets.MNIST`")
            download_successful = False
        _TORCHVISION_AVAILABLE &= download_successful

        if _TORCHVISION_AVAILABLE and should_raise:
            raise MisconfigurationException(
                "The `torchvision` package is available. This implementation is meant for internal use."
                " Please import MNIST with `from torchvision.datasets import MNIST`"
                " instead of `from pytorch_lightning.utilities.debug_examples import MNIST`"
            )
        return _TORCHVISION_AVAILABLE
