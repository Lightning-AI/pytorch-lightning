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
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from pytorch_lightning import Callback, seed_everything, Trainer
from pytorch_lightning.accelerators import CPUAccelerator, HPUAccelerator
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins import HPUPlugin, HPUPrecisionPlugin
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import _HPU_AVAILABLE, DeviceType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel


class HPUModel(BoringModel):
    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def training_epoch_end(self, outputs) -> None:
        pass

    def validation_epoch_end(self, outputs) -> None:
        pass

    def test_epoch_end(self, outputs) -> None:
        pass


@pytest.mark.skipif(_HPU_AVAILABLE, reason="test requires non-HPU machine")
def test_fail_if_no_hpus(tmpdir):
    with pytest.raises(MisconfigurationException, match="HPU Accelerator requires HPU devices to run"):
        Trainer(default_root_dir=tmpdir, hpus=1)

    with pytest.raises(MisconfigurationException, match="HPU Accelerator requires HPU devices to run"):
        Trainer(default_root_dir=tmpdir, hpus=1, accelerator="hpu")


@RunIf(hpu=True)
def test_accelerator_selected(tmpdir):
    trainer = Trainer(default_root_dir=tmpdir, hpus=1)
    assert isinstance(trainer.accelerator, HPUAccelerator)
    trainer = Trainer(default_root_dir=tmpdir, hpus=1, accelerator="hpu")
    assert isinstance(trainer.accelerator, HPUAccelerator)


@RunIf(hpu=True)
def test_warning_if_hpus_not_used(tmpdir):
    with pytest.warns(UserWarning, match="HPU available but not used. Set the `hpus` flag in your trainer"):
        Trainer(default_root_dir=tmpdir)

## TBD
@pytest.mark.skipif(_HPU_AVAILABLE, reason="PyTorch is not linked with support for hpu devices")
@RunIf(hpu=True)
@pytest.mark.parametrize("hpus", [1, 8])
def test_all_stages(tmpdir, hpus):
    if hpus > 1:
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "hcl"    
    model = HPUModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, hpus=hpus)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


@RunIf(hpu=True)
def test_mixed_precision(tmpdir):
    class TestCallback(Callback):
        def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
            assert trainer.accelerator.model.precision == 16
            raise SystemExit

    hmp_keys = ["level", "verbose", "bf16_ops", "fp32_ops"]
    hmp_params = dict.fromkeys(hmp_keys)
    hmp_params["level"] = "O1"
    hmp_params["verbose"] = False
    hmp_params["bf16_ops"] = "./pytorch-lightning-fork/pl_examples/hpu_examples/simple_mnist/ops_bf16_mnist.txt"
    hmp_params["fp32_ops"] = "./pytorch-lightning-fork/pl_examples/hpu_examples/simple_mnist/ops_fp32_mnist.txt"

    model = HPUModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, hpus=1, precision=16, hmp_params=hmp_params, callbacks=TestCallback())
    assert isinstance(trainer.accelerator.precision_plugin, HPUPrecisionPlugin)
    assert trainer.accelerator.precision_plugin.precision == 16
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(hpu=True)
def test_precision_plugin(tmpdir):
    """Ensure precision plugin value is set correctly."""

    hmp_keys = ["level", "verbose", "bf16_ops", "fp32_ops"]
    hmp_params = dict.fromkeys(hmp_keys)
    hmp_params["level"] = "O1"
    hmp_params["verbose"] = False
    hmp_params["bf16_ops"] = "./pytorch-lightning-fork/pl_examples/hpu_examples/simple_mnist/ops_bf16_mnist.txt"
    hmp_params["fp32_ops"] = "./pytorch-lightning-fork/pl_examples/hpu_examples/simple_mnist/ops_fp32_mnist.txt"

    plugin = HPUPrecisionPlugin(precision=16, hmp_params=hmp_params)
    assert plugin.precision == 16


@RunIf(hpu=True)
def test_accelerator_hpu():

    trainer = Trainer(accelerator="hpu", hpus=1)

    assert trainer._device_type == "hpu"
    assert isinstance(trainer.accelerator, HPUAccelerator)

    with pytest.raises(
        MisconfigurationException, match="You passed `accelerator='hpu'`, but you didn't pass `hpus` to `Trainer`"
    ):
        trainer = Trainer(accelerator="hpu")

    trainer = Trainer(accelerator="auto", hpus=8)

    assert trainer._device_type == "hpu"
    assert isinstance(trainer.accelerator, HPUAccelerator)


@RunIf(hpu=True)
def test_accelerator_cpu_with_hpus_flag():

    trainer = Trainer(accelerator="cpu", hpus=1)

    assert trainer._device_type == "cpu"
    assert isinstance(trainer.accelerator, CPUAccelerator)


@RunIf(hpu=True)
def test_accelerator_hpu_with_devices():

    trainer = Trainer(accelerator="hpu", devices=1)

    assert trainer.hpus == 1
    assert isinstance(trainer.training_type_plugin, HPUPlugin)
    assert isinstance(trainer.accelerator, HPUAccelerator)


@RunIf(hpu=True)
def test_accelerator_auto_with_devices_hpu():

    trainer = Trainer(accelerator="auto", devices=1)

    assert trainer._device_type == "hpu"
    assert trainer.hpus == 1
