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
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from pytorch_lightning import LightningDataModule, LightningModule


class RandomDictDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        a = self.data[index]
        b = a + 2
        return {'a': a, 'b': b}

    def __len__(self):
        return self.len


class RandomDictStringDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return {"id": str(index), "x": self.data[index]}

    def __len__(self):
        return self.len


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class RandomIterableDataset(IterableDataset):

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self):
        return self.count


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
        torch.stack([x['x'] for x in outputs]).mean()

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

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None
        self.checkpoint_state: Optional[str] = None

    def prepare_data(self):
        self.random_full = RandomDataset(32, 192)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.random_train = Subset(self.random_full, indices=range(64))
            self.dims = self.random_train[0].shape

        if stage in ("fit", "validate") or stage is None:
            self.random_val = Subset(self.random_full, indices=range(64, 128))

        if stage == "test" or stage is None:
            self.random_test = Subset(self.random_full, indices=range(128, 192))
            self.dims = getattr(self, "dims", self.random_test[0].shape)

    def train_dataloader(self):
        return DataLoader(self.random_train)

    def val_dataloader(self):
        return DataLoader(self.random_val)

    def test_dataloader(self):
        return DataLoader(self.random_test)
