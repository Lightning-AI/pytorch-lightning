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

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
from torch.nn import functional as F


class LitClassifier(LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

        self.val_outptus = []
        self.test_outputs = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        self.val_outputs.append(acc)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.test_outputs.append(acc)
        return acc

    def accuracy(self, logits, y):
        # currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        return torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)

    def on_validation_epoch_end(self) -> None:
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        self.log("val_acc", torch.stack(self.val_outptus).mean(), prog_bar=True)
        self.val_outptus.clear()

    def on_test_epoch_end(self) -> None:
        self.log("test_acc", torch.stack(self.test_outputs).mean())
        self.test_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    dm = MNISTDataModule(batch_size=32)
    model = LitClassifier()
    trainer = Trainer(max_epochs=2, accelerator="ipu", devices=8)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
