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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy, MeanSquaredError
from tests.helpers.datasets import ExampleDataset


class ClassificationModel(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()

        self.lr = lr
        for i in range(3):
            setattr(self, f"layer_{i}", nn.Linear(32, 32))
            setattr(self, f"layer_{i}a", torch.nn.ReLU())
        setattr(self, "layer_end", nn.Linear(32, 3))

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        x = self.layer_end(x)
        logits = F.softmax(x, dim=1)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("val_loss", F.cross_entropy(logits, y), prog_bar=False)
        self.log("val_acc", self.valid_acc(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("test_loss", F.cross_entropy(logits, y), prog_bar=False)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True)


class RegressionModel(LightningModule):
    def __init__(self):
        super().__init__()
        setattr(self, "layer_0", nn.Linear(16, 64))
        setattr(self, "layer_0a", torch.nn.ReLU())
        for i in range(1, 3):
            setattr(self, f"layer_{i}", nn.Linear(64, 64))
            setattr(self, f"layer_{i}a", torch.nn.ReLU())
        setattr(self, "layer_end", nn.Linear(64, 1))

        self.train_mse = MeanSquaredError()
        self.valid_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self._example_input_array = torch.randn(1, 16)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        x = self.layer_end(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_MSE", self.train_mse(out, y), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        self.log("val_loss", F.mse_loss(out, y), prog_bar=False)
        self.log("val_MSE", self.valid_mse(out, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        self.log("test_loss", F.mse_loss(out, y), prog_bar=False)
        self.log("test_MSE", self.test_mse(out, y), prog_bar=True)

    @property
    def example_input_array(self):
        return self._example_input_array

    @example_input_array.setter
    def example_input_array(self, ex):
        """Need a way to set to toggle None for test_quantization_exceptions"""
        self._example_input_array = ex


class MultiInputModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3)
        self.ff = nn.quantized.FloatFunctional()
        self.triggered = False

    def forward(self, a, b, trigger=False):
        """Use 2 positional arguments and 1 keyword argument"""
        feat_a = self.conv(a)
        feat_b = self.conv(b)
        combined = self.ff.add(feat_a, feat_b)
        if trigger:
            self.triggered = True
        return combined

    def training_step(self, batch, batch_idx):
        """Dummy loss as-in BoringModel"""
        out = self.forward(*batch)
        return torch.nn.functional.mse_loss(out, torch.ones_like(out))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @property
    def example_input_array(self):
        """Two MNIST-sized squares"""
        return torch.randn(1, 3, 28, 28), torch.randn(1, 3, 28, 28)


class MultiOutputModel(LightningModule):
    def __init__(self, output_dtype: type):
        super().__init__()
        if output_dtype not in (list, tuple):
            raise ValueError("output_dtype must be one of (list, tuple)")
        self.output_dtype = output_dtype
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        return self.output_dtype([c1, c2])

    def training_step(self, batch, batch_idx):
        """Dummy loss as-in BoringModel"""
        out = self.forward(batch)
        loss = torch.tensor(0.0)
        for o in out:
            loss += torch.nn.functional.mse_loss(o, torch.ones_like(o))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(ExampleDataset(self), batch_size=2)

    @property
    def example_input_array(self):
        """MNIST-size square image"""
        return torch.randn(1, 3, 28, 28)
