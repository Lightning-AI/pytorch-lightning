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

from argparse import ArgumentParser
from pprint import pprint

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule
from pytorch_lightning.accelerators import IPUAccelerator


class Block(nn.Module):

    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class LitClassifier(pl.LightningModule):

    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.layer1 = Block(1, 32, 3, 2)
        self.layer2 = Block(32, 64, 3, 2)
        self.layer3 = nn.Linear(1600, 128)
        self.layer3_act = nn.ReLU()
        self.layer3_dropout = torch.nn.Dropout(0.5)
        self.layer4 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # Flatten layer
        x = x.view(-1, 1600)
        x = self.layer3_act(self.layer3(x))
        x = self.layer4(self.layer3_dropout(x))
        x = self.softmax(x)
        return x

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    parser = IPUAccelerator.add_argparse_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    dm = MNISTDataModule.from_argparse_args(args)

    model = LitClassifier(args.learning_rate)

    accelerator = IPUAccelerator.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args, accelerator=accelerator)

    trainer.fit(model, datamodule=dm)

    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()
