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

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel


def test_v2_0_0_unsupported_datamodule_on_save_load_checkpoint():
    datamodule = LightningDataModule()
    with pytest.raises(
        NotImplementedError,
        match="`LightningDataModule.on_save_checkpoint`.*no longer supported as of v1.8",
    ):
        datamodule.on_save_checkpoint({})

    with pytest.raises(
        NotImplementedError,
        match="`LightningDataModule.on_load_checkpoint.*no longer supported as of v1.8",
    ):
        datamodule.on_load_checkpoint({})

    class OnSaveDataModule(BoringDataModule):
        def on_save_checkpoint(self, checkpoint):
            pass

    class OnLoadDataModule(BoringDataModule):
        def on_load_checkpoint(self, checkpoint):
            pass

    trainer = Trainer()
    model = BoringModel()

    with pytest.raises(
        NotImplementedError,
        match="`LightningDataModule.on_save_checkpoint`.*no longer supported as of v1.8.",
    ):
        trainer.fit(model, OnSaveDataModule())

    with pytest.raises(
        NotImplementedError,
        match="`LightningDataModule.on_load_checkpoint`.*no longer supported as of v1.8.",
    ):
        trainer.fit(model, OnLoadDataModule())


def test_v2_0_0_lightning_module_unsupported_use_amp():
    model = BoringModel()
    with pytest.raises(
        RuntimeError,
        match="`LightningModule.use_amp`.*no longer accessible as of v1.8.",
    ):
        model.use_amp

    with pytest.raises(
        RuntimeError,
        match="`LightningModule.use_amp`.*no longer accessible as of v1.8.",
    ):
        model.use_amp = False
