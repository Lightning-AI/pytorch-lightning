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
import pytest

import os
import collections
import pytest
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, callbacks

from tests.base.boring_model import BoringModel, RandomDictDataset, RandomDictStringDataset
from tests.base.deterministic_model import DeterministicModel

from pytorch_lightning.trainer import Trainer
from time import sleep


def test_sleep(tmpdir):
    for i in range(300):
        sleep(1)
        print(i)


def test_multi_node_gpu(tmpdir):

    class TestModel(BoringModel):
        def on_train_epoch_start(self) -> None:
            print("override any method to prove your bug")

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("val_loss", loss)

    # model
    model = TestModel()
    model.validation_epoch_end = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=8,
        limit_val_batches=8,
        max_epochs=2,
        weights_summary=None,
        num_nodes=2,
        gpus=1,

    )
    trainer.fit(model)




