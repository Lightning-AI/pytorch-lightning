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
from contextlib import suppress
from typing import Optional

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Callback, LightningDataModule, Trainer
from pytorch_lightning.utilities.distributed import distributed_available
from tests.helpers.boring_model import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


class TestModel(BoringModel):
    def setup(self, stage: Optional[str]) -> None:
        self.has_setup = distributed_available()


class PlDataModule(LightningDataModule):
    def prepare_data(self):
        self._has_prepared = distributed_available()

    def setup(self, stage: Optional[str]) -> None:
        self._setup = distributed_available()

    def _check(self):
        has_prepared = getattr(self, "_has_prepared", None)
        if self.trainer.is_global_zero:
            assert not has_prepared
        else:
            assert has_prepared is None
        assert self._setup

    def train_dataloader(self):
        self._check()
        assert self._setup
        return DataLoader(RandomDataset(32, 64), batch_size=2)

    def val_dataloader(self):
        self._check()
        assert self._setup
        return DataLoader(RandomDataset(32, 64), batch_size=2)


def _test_has_prepared_and_setup(tmpdir, plugins):
    model = TestModel()
    dm = PlDataModule()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, gpus=2, plugins=plugins, limit_train_batches=1, limit_val_batches=1
    )
    trainer.fit(model, dm)


@RunIf(min_gpus=2, special=True)
def test_has_prepared_and_setup_ddp(tmpdir):
    _test_has_prepared_and_setup(tmpdir, "ddp")


@RunIf(min_gpus=2, special=True)
def test_has_prepared_and_setup_ddp_sharded(tmpdir):
    _test_has_prepared_and_setup(tmpdir, "ddp_sharded")


@RunIf(min_gpus=2, special=True)
def test_has_prepared_and_setup_deepseed(tmpdir):
    _test_has_prepared_and_setup(tmpdir, "deepspeed")


@RunIf(min_gpus=2, special=True)
def test_has_prepared_and_setup_ddp_spawn(tmpdir):
    _test_has_prepared_and_setup(tmpdir, "ddp_spawn")
