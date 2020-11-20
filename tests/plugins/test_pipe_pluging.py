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
from torch import nn

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.fairscale_pipe_pluging import PipePlugin
from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from tests.base.boring_model import BoringModel


class SequentialModel(BoringModel):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        output = self.layer(batch)
        if self.final_stage:
            loss = self.loss(batch, output)
            self.manual_backward(loss, opt)
            self.manual_optimizer_step(opt)
        else:
            self.back_helper(output)


def test_pipe_plugin(tmpdir):

    model = SequentialModel()
    model.training_step_end = None
    model.training_epoch_end = None
    trainer = Trainer(
        fast_dev_run=True,
        gpus=0,
        distributed_backend='ddp_cpu',
        plugins=[PipePlugin()],
        automatic_optimization=False,
    )
    trainer.fit(model)
