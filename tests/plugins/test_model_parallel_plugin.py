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
from distutils.version import LooseVersion
from unittest import mock

import fairscale
import pytest
import torch
import torch.distributed as torch_distrib
from torch import nn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.model_parallel_plugin import ModelParallelPlugin
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from tests.backends.launcher import DDPLauncher
from tests.base.boring_model import BoringModel, RandomDataset


class SequentialModel(LightningModule):

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
        self.layers = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.layer(x)

    def loss(self, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        output = self.layers(batch)
        if self.final_stage:
            loss = self.loss(output)
            self.manual_backward(loss, opt)
            self.manual_optimizer_step(opt)
            self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        else:
            self.back_helper(output)

    def validation_step(self, batch, batch_idx):
        output = self.layers(batch)
        if self.final_stage:
            loss = self.loss(output)
            self.log("val_loss", loss, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        if self.final_stage:
            loss = self.loss(batch, output)
            self.log("test_loss", loss, on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layers.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@DDPLauncher.run("--distributed_backend ddp --gpus 2")
def test_model_parallel_plugin(tmpdir, args=None):

    model = SequentialModel()
    model.training_step_end = None
    model.training_epoch_end = None
    model.validation_epoch_end = None
    trainer = Trainer(
        gpus=args.gpus,
        distributed_backend=args.distributed_backend,
        plugins=[ModelParallelPlugin(balance=[2, 1])],
        automatic_optimization=False,
    )
    trainer.fit(model)

    torch_distrib.rpc.shutdown()
