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
from jsonargparse import lazy_instance
from torch.nn import functional as F

import pytorch_lightning as pl
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule
from pytorch_lightning.plugins import HPUPrecisionPlugin
from pytorch_lightning.utilities.cli import LightningCLI


class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log("test_acc", acc)

    @staticmethod
    def accuracy(logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    cli = LightningCLI(
        LitClassifier,
        MNISTDataModule,
        trainer_defaults={
            "accelerator": "hpu",
            "devices": 1,
            "max_epochs": 1,
            "plugins": lazy_instance(HPUPrecisionPlugin, precision=16),
        },
        run=False,
        save_config_overwrite=True,
    )

    # Run the model ⚡
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.validate(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)
