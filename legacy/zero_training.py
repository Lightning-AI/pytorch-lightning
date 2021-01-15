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
import os

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl

PATH_LEGACY = os.path.dirname(__file__)


class RandomDataset(Dataset):
    def __init__(self, size, length: int = 100):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class DummyModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def _loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def _step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self._loss(batch, output)
        # return {'loss': loss}  # used for PL<1.0
        return loss  # used for PL >= 1.0

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


def main_train(dir_path, max_epochs: int = 5):

    trainer = pl.Trainer(
        default_root_dir=dir_path,
        checkpoint_callback=True,
        max_epochs=max_epochs,
    )

    model = DummyModel()
    trainer.fit(model)


if __name__ == '__main__':
    path_dir = os.path.join(PATH_LEGACY, 'checkpoints', str(pl.__version__))
    main_train(path_dir)
