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

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# USE THIS MODEL TO REPRODUCE A BUG YOU REPORT
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
import os

import torch
from torch.utils.data import Dataset

from pl_examples import cli_lightning_logo
from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):
    """
    >>> RandomDataset(size=10, length=20)  # doctest: +ELLIPSIS
    <...bug_report_model.RandomDataset object at ...>
    """

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    """
    >>> BoringModel()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    BoringModel(
      (layer): Linear(...)
    )
    """

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
        x = self.layer(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x['x'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


#  NOTE: If you are using a cmd line to run your script,
#  provide the cmd line as below.
#  opt = "--max_epochs 1 --limit_train_batches 1".split(" ")
#  parser = ArgumentParser()
#  args = parser.parse_args(opt)


def test_run():

    class TestModel(BoringModel):

        def on_train_epoch_start(self) -> None:
            print('override any method to prove your bug')

    # fake data
    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64))
    val_data = torch.utils.data.DataLoader(RandomDataset(32, 64))
    test_data = torch.utils.data.DataLoader(RandomDataset(32, 64))

    # model
    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model, train_data, val_data)
    trainer.test(test_dataloaders=test_data)


if __name__ == '__main__':
    cli_lightning_logo()
    test_run()
