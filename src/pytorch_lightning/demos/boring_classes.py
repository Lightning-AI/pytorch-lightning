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
from typing import cast, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


class RandomDictDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

    def __len__(self) -> int:
        return self.len


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self) -> int:
        return self.count


class BoringModel(LightningModule):
    def __init__(self) -> None:
        """Testing PL Module.

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

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def loss(self, batch: Tensor, preds: Tensor) -> Tensor:
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(preds, torch.ones_like(preds))

    def step(self, x: Tensor) -> Tensor:
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        return training_step_outputs

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = cast(List[Dict[str, Tensor]], outputs)
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = cast(List[Dict[str, Tensor]], outputs)
        torch.stack([x["x"] for x in outputs]).mean()

    def test_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = cast(List[Dict[str, Tensor]], outputs)
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[_LRScheduler]]:
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
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
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None
        self.checkpoint_state: Optional[str] = None
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


class ManualOptimBoringModel(BoringModel):
    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        opt = self.optimizers()
        assert isinstance(opt, (Optimizer, LightningOptimizer))
        output = self(batch)
        loss = self.loss(batch, output)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss


class DemoModel(LightningModule):
    def __init__(self, out_dim: int = 10, learning_rate: float = 0.02):
        super().__init__()
        self.l1 = torch.nn.Linear(32, out_dim)
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch: Tensor, batch_nb: int) -> STEP_OUTPUT:
        x = batch
        x = self(x)
        loss = x.sum()
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Net(nn.Module):
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
        output = F.log_softmax(x, dim=1)
        return output
