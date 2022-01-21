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
import torch.nn.functional as F
from torch import nn
from torchmetrics import MeanSquaredError
from torchmetrics.functional import mean_absolute_percentage_error as mape

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks.intel_compressor import INCQuantization
from pytorch_lightning.utilities.memory import get_model_size_mb
from tests.helpers.datamodules import RegressDataModule
from tests.helpers.runif import RunIf


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = nn.Linear(16, 64)
        self.layer_0a = torch.nn.ReLU()
        self.layer_1 = nn.Linear(64, 64)
        self.layer_1a = torch.nn.ReLU()
        self.layer_2 = nn.Linear(64, 64)
        self.layer_2a = torch.nn.ReLU()
        self.layer_end = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        x = self.layer_end(x)
        return x


class LightningModel(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()

        self.lr = lr
        self.model = M()

        self.train_mse = MeanSquaredError()
        self.valid_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x):
        x = self.model(x)
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


def build_yaml():
    fake_ptq_yaml = """
        version: 1.0

        model:
          name: imagenet
          framework: pytorch_fx

        device: cpu

        quantization:
          approach: post_training_static_quant
          calibration:
            sampling_size: 800

        tuning:
          accuracy_criterion:
            relative:  0.3
            higher_is_better: False
          exit_policy:
            timeout: 0
            max_trials: 1200
          random_seed: 9527
    """
    with open("ptq_yaml.yaml", "w", encoding="utf-8") as f:
        f.write(fake_ptq_yaml)


def tearDown():
    os.remove("ptq_yaml.yaml")


@RunIf(quantization=True)
def test_INCQuantization(tmpdir):
    build_yaml()

    seed_everything(42)
    dm = RegressDataModule()
    model = LightningModel()

    trainer_args = dict(default_root_dir=tmpdir, max_epochs=7, gpus=int(torch.cuda.is_available()))
    trainer = Trainer(**trainer_args)
    trainer.fit(model, datamodule=dm)
    org_size = get_model_size_mb(model)
    org_score = torch.mean(torch.tensor([mape(model(x), y) for x, y in dm.test_dataloader()]))

    inc_cb = INCQuantization(
        "ptq_yaml.yaml", monitor="val_MSE", module_name_to_quant="model", datamodule=dm, dirpath=tmpdir
    )
    trainer = Trainer(callbacks=[inc_cb], **trainer_args)
    trainer.fit(model, datamodule=dm)

    tearDown()

    quant_score = torch.mean(torch.tensor([mape(model(x), y) for x, y in dm.test_dataloader()]))
    quant_size = get_model_size_mb(model)
    assert torch.allclose(org_score, quant_score, atol=0.45)
    size_ratio = quant_size / org_size
    assert size_ratio < 0.65
