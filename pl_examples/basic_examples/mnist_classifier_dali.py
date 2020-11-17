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
from argparse import ArgumentParser
from random import shuffle
from warnings import warn

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import random_split

import pytorch_lightning as pl

try:
    from torchvision import transforms
    from torchvision.datasets.mnist import MNIST
except Exception:
    from tests.base.datasets import MNIST

try:
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
except (ImportError, ModuleNotFoundError):
    warn('NVIDIA DALI is not available')
    ops, types, Pipeline, DALIClassificationIterator = ..., ..., ABC, ABC


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
    This class extends DALI's original DALIClassificationIterator with the __len__() function so that we can call len() on it
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
        super().__init__(pipelines, size, reader_name, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded)

    def __len__(self):
        batch_count = self._size // (self._num_gpus * self.batch_size)
        last_batch = 1 if self._fill_last_batch else 0
        return batch_count + last_batch


class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    eii_train = ExternalMNISTInputIterator(mnist_train, args.batch_size)
    eii_val = ExternalMNISTInputIterator(mnist_val, args.batch_size)
    eii_test = ExternalMNISTInputIterator(mnist_test, args.batch_size)

    pipe_train = ExternalSourcePipeline(batch_size=args.batch_size, eii=eii_train, num_threads=2, device_id=0)
    train_loader = DALIClassificationLoader(pipe_train, size=len(mnist_train), auto_reset=True, fill_last_batch=False)

    pipe_val = ExternalSourcePipeline(batch_size=args.batch_size, eii=eii_val, num_threads=2, device_id=0)
    val_loader = DALIClassificationLoader(pipe_val, size=len(mnist_val), auto_reset=True, fill_last_batch=False)

    pipe_test = ExternalSourcePipeline(batch_size=args.batch_size, eii=eii_test, num_threads=2, device_id=0)
    test_loader = DALIClassificationLoader(pipe_test, size=len(mnist_test), auto_reset=True, fill_last_batch=False)

    # ------------
    # model
    # ------------
    model = LitClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
