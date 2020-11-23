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
from torch.utils.data import Dataset

from pytorch_lightning import LightningModule


class RandomDictDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

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


class BoringModel(LightningModule):
    def __init__(
        self,
        batch_size: int = 1,
        in_features: int = 32,
        learning_rate: float = 0.1,
        optimizer_name: str = "SGD",
        out_features: int = 2,
    ):
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
        self.layer = torch.nn.Linear(in_features, out_features)
        self.batch_size = batch_size
        self.in_features = in_features
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.out_features = out_features

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
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.layer.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    def get_default_hparams(continue_training: bool = False, hpc_exp_number: int = 0) -> dict:
        args = dict(
            batch_size=1,
            in_features=32,
            learning_rate=0.1,
            optimizer_name="SGD",
            out_features=2,
        )

        return args
