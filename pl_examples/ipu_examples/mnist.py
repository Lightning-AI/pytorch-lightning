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
from torch.nn import functional as F

import pytorch_lightning as pl
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule


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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.log('val_loss', outputs.mean(), prog_bar=True)

    def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.log('test_loss', outputs.mean(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == '__main__':
    dm = MNISTDataModule(batch_size=32)

    model = LitClassifier()

    trainer = pl.Trainer(max_epochs=10, accelerator='ipu', ipu_cores=8)

    trainer.fit(model, datamodule=dm)

    result = trainer.test(model, datamodule=dm)
    pprint(result)
