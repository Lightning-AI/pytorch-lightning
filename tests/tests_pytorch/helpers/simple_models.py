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
import operator

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning_utilities.core.imports import compare_version
from torch import nn
from torchmetrics import Accuracy, MeanSquaredError

# using new API with task
_TM_GE_0_11 = compare_version("torchmetrics", operator.ge, "0.11.0")


class ClassificationModel(LightningModule):
    def __init__(self, num_features=32, num_classes=3, lr=0.01):
        super().__init__()

        self.lr = lr
        for i in range(3):
            setattr(self, f"layer_{i}", nn.Linear(num_features, num_features))
            setattr(self, f"layer_{i}a", torch.nn.ReLU())
        setattr(self, "layer_end", nn.Linear(num_features, 3))

        acc = Accuracy(task="multiclass", num_classes=num_classes) if _TM_GE_0_11 else Accuracy()
        self.train_acc = acc.clone()
        self.valid_acc = acc.clone()
        self.test_acc = acc.clone()

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        x = self.layer_end(x)
        return F.softmax(x, dim=1)

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

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.forward(x)


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
