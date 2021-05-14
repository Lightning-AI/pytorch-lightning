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
from abc import ABC
from random import shuffle
from warnings import warn

import numpy as np
import torch
from packaging.version import Version
from torch.nn import functional as F
from torch.utils.data import random_split

import pytorch_lightning as pl
from pl_examples import _DALI_AVAILABLE, _DATASETS_PATH, _TORCHVISION_MNIST_AVAILABLE, cli_lightning_logo
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
if _TORCHVISION_MNIST_AVAILABLE:
    from torchvision.datasets import MNIST
else:
    from tests.helpers.datasets import MNIST

if _DALI_AVAILABLE:
    from nvidia.dali import __version__ as dali_version
    from nvidia.dali import ops
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    NEW_DALI_API = Version(dali_version) >= Version('0.28.0')
    if NEW_DALI_API:
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy
else:
    warn('NVIDIA DALI is not available')
    ops, Pipeline, DALIClassificationIterator, LastBatchPolicy = ..., ABC, ABC, ABC


class ExternalMNISTInputIterator(object):
    """
    This iterator class wraps torchvision's MNIST dataset and returns the images and labels in batches
    """

    def __init__(self, mnist_ds, batch_size):
        self.batch_size = batch_size
        self.mnist_ds = mnist_ds
        self.indices = list(range(len(self.mnist_ds)))
        shuffle(self.indices)

    def __iter__(self):
        self.i = 0
        self.n = len(self.mnist_ds)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            index = self.indices[self.i]
            img, label = self.mnist_ds[index]
            batch.append(img.numpy())
            labels.append(np.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)


class ExternalSourcePipeline(Pipeline):
    """
    This DALI pipeline class just contains the MNIST iterator
    """

    def __init__(self, batch_size, eii, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.source = ops.ExternalSource(source=eii, num_outputs=2)
        self.build()

    def define_graph(self):
        images, labels = self.source()
        return images, labels


class DALIClassificationLoader(DALIClassificationIterator):
    """
    This class extends DALI's original `DALIClassificationIterator` with the `__len__()` function
     so that we can call `len()` on it
    """

    def __init__(
        self,
        pipelines,
        size=-1,
        reader_name=None,
        auto_reset=False,
        fill_last_batch=True,
        dynamic_shape=False,
        last_batch_padded=False,
    ):
        if NEW_DALI_API:
            last_batch_policy = LastBatchPolicy.FILL if fill_last_batch else LastBatchPolicy.DROP
            super().__init__(
                pipelines,
                size,
                reader_name,
                auto_reset,
                dynamic_shape,
                last_batch_policy=last_batch_policy,
                last_batch_padded=last_batch_padded
            )
        else:
            super().__init__(
                pipelines, size, reader_name, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded
            )
        self._fill_last_batch = fill_last_batch

    def __len__(self):
        batch_count = self._size // (self._num_gpus * self.batch_size)
        last_batch = 1 if self._fill_last_batch else 1
        return batch_count + last_batch


class LitClassifier(pl.LightningModule):

    def __init__(
        self,
        hidden_dim: int = 128,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def split_batch(self, batch):
        return batch[0]["data"], batch[0]["label"].squeeze().long()

    def training_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 32,
    ):
        super().__init__()
        dataset = MNIST(_DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(_DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

        eii_train = ExternalMNISTInputIterator(self.mnist_train, batch_size)
        eii_val = ExternalMNISTInputIterator(self.mnist_val, batch_size)
        eii_test = ExternalMNISTInputIterator(self.mnist_test, batch_size)

        self.pipe_train = ExternalSourcePipeline(batch_size=batch_size, eii=eii_train, num_threads=2, device_id=0)
        self.pipe_val = ExternalSourcePipeline(batch_size=batch_size, eii=eii_val, num_threads=2, device_id=0)
        self.pipe_test = ExternalSourcePipeline(batch_size=batch_size, eii=eii_test, num_threads=2, device_id=0)

    def train_dataloader(self):
        return DALIClassificationLoader(
            self.pipe_train,
            size=len(self.mnist_train),
            auto_reset=True,
            fill_last_batch=True,
        )

    def val_dataloader(self):
        return DALIClassificationLoader(
            self.pipe_val,
            size=len(self.mnist_val),
            auto_reset=True,
            fill_last_batch=False,
        )

    def test_dataloader(self):
        return DALIClassificationLoader(
            self.pipe_test,
            size=len(self.mnist_test),
            auto_reset=True,
            fill_last_batch=False,
        )


def cli_main():
    if not _DALI_AVAILABLE:
        return

    cli = LightningCLI(LitClassifier, MyDataModule, seed_everything_default=1234)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
