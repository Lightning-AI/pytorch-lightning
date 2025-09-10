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
from collections.abc import Iterable, Iterator
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT


class RandomDictDataset(Dataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

    def __len__(self) -> int:
        return self.len


class RandomDataset(Dataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class RandomIterableDataset(IterableDataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self) -> int:
        return self.count


class BoringModel(LightningModule):
    """Testing PL Module.

    Use as follows:
    - subclass
    - modify the behavior for what you want

    .. warning::  This is meant for testing/debugging and is experimental.

    Example::

        class TestModel(BoringModel):
            def training_step(self, ...):
                ...  # do your own thing

    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def loss(self, preds: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if labels is None:
            labels = torch.ones_like(preds)
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(preds, labels)

    def step(self, batch: Any) -> Tensor:
        output = self(batch)
        return self.loss(output)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return {"loss": self.step(batch)}

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return {"x": self.step(batch)}

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return {"y": self.step(batch)}

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[LRScheduler]]:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))


class BoringDataModule(LightningDataModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        super().__init__()
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.random_train = Subset(self.random_full, indices=range(64))

        if stage in ("fit", "validate"):
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))

        if stage == "test":
            self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))

        if stage == "predict":
            self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.random_train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.random_val)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.random_test)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.random_predict)


class BoringDataModuleNoLen(LightningDataModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        super().__init__()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.random_train = RandomIterableDataset(32, 512)

        if stage in ("fit", "validate"):
            self.random_val = RandomIterableDataset(32, 128)

        if stage == "test":
            self.random_test = RandomIterableDataset(32, 256)

        if stage == "predict":
            self.random_predict = RandomIterableDataset(32, 64)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.random_train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.random_val)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.random_test)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.random_predict)


class IterableBoringDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_datasets = [
                RandomDataset(4, 16),
                RandomIterableDataset(4, 16),
            ]

        if stage in ("fit", "validate"):
            self.val_datasets = [
                RandomDataset(4, 32),
                RandomIterableDataset(4, 32),
            ]

        if stage == "test":
            self.test_datasets = [
                RandomDataset(4, 64),
                RandomIterableDataset(4, 64),
            ]

        if stage == "predict":
            self.predict_datasets = [
                RandomDataset(4, 128),
                RandomIterableDataset(4, 128),
            ]

    def train_dataloader(self) -> Iterable[DataLoader]:
        combined_train = apply_to_collection(self.train_datasets, Dataset, lambda x: DataLoader(x))
        return combined_train

    def val_dataloader(self) -> DataLoader:
        combined_val = apply_to_collection(self.val_datasets, Dataset, lambda x: DataLoader(x))
        return combined_val

    def test_dataloader(self) -> DataLoader:
        combined_test = apply_to_collection(self.test_datasets, Dataset, lambda x: DataLoader(x))
        return combined_test

    def predict_dataloader(self) -> DataLoader:
        combined_predict = apply_to_collection(self.predict_datasets, Dataset, lambda x: DataLoader(x))
        return combined_predict


class ManualOptimBoringModel(BoringModel):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        opt = self.optimizers()
        assert isinstance(opt, (Optimizer, LightningOptimizer))
        loss = self.step(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss


class DemoModel(LightningModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, out_dim: int = 10, learning_rate: float = 0.02):
        super().__init__()
        self.l1 = torch.nn.Linear(32, out_dim)
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch: Any, batch_nb: int) -> STEP_OUTPUT:
        x = batch
        x = self(x)
        return x.sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Net(nn.Module):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
