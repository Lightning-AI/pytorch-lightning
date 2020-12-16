#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger

import os
from typing import Any, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import torch
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam


class MNISTModule(LightningModule):

    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=1e-3)
        return opt



# noinspection PyAttributeOutsideInit
class MNISTDataModule(LightningDataModule):

    def __init__(self):
        super().__init__()
        self.train_dims = None
        self.vocab_size = 0

    def prepare_data(self):
        # called only on 1 GPU
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        self.test = MNIST(os.getcwd(), train=False, download=False, transform=transform)

        self.train, self.val = torch.utils.data.random_split(self.train, (50000, 10000))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, shuffle=True, drop_last=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=512, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=512, drop_last=False)


if __name__ == '__main__':
    dm = MNISTDataModule()
    model = MNISTModule()

    params = dict(param1='a', param2=1)
    trainer = Trainer(gpus=2, max_epochs=1, accelerator='ddp')
    trainer.fit(model, datamodule=dm)

    print(trainer.global_rank, trainer.logger.version)

    result = trainer.test()
    print(result)