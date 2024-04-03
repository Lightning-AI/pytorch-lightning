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
import os
from functools import partial
from unittest import mock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import XLAAccelerator
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.strategies import XLAStrategy
from lightning.pytorch.strategies.launchers.xla import _XLALauncher
from lightning.pytorch.trainer.connectors.logger_connector.result import _Sync
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

import tests_pytorch.helpers.pipelines as tpipes
from tests_pytorch.helpers.runif import RunIf


class SerialLoaderBoringModel(BoringModel):
    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 2000), batch_size=32)

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 2000), batch_size=32)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_devices_1(tmp_path):
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 2,
        "accelerator": "tpu",
        "devices": 1,
        "limit_train_batches": 3,
        "limit_val_batches": 3,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)


@pytest.mark.parametrize("tpu_core", [1, 3])
@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_index(tmp_path, tpu_core):
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 2,
        "accelerator": "tpu",
        "devices": [tpu_core],
        "limit_train_batches": 3,
        "limit_val_batches": 3,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)
    import torch_xla

    assert torch_xla._XLAC._xla_get_default_device() == f"xla:{tpu_core}"


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_multiple_tpu_devices(tmp_path):
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "accelerator": "tpu",
        "devices": "auto",
        "limit_train_batches": 3,
        "limit_val_batches": 3,
    }

    # multiple cores needs a big dataset
    model = SerialLoaderBoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False, min_acc=0.05)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_16bit_tpu_devices_1(tmp_path):
    trainer_options = {
        "default_root_dir": tmp_path,
        "precision": "16-true",
        "enable_progress_bar": False,
        "max_epochs": 2,
        "accelerator": "tpu",
        "devices": 1,
        "limit_train_batches": 3,
        "limit_val_batches": 2,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.parametrize("tpu_core", [1, 3])
@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_16bit_tpu_index(tmp_path, tpu_core):
    trainer_options = {
        "default_root_dir": tmp_path,
        "precision": "16-true",
        "enable_progress_bar": False,
        "max_epochs": 2,
        "accelerator": "tpu",
        "devices": [tpu_core],
        "limit_train_batches": 3,
        "limit_val_batches": 2,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)
    import torch_xla

    assert torch_xla._XLAC._xla_get_default_device() == f"xla:{tpu_core}"


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_16bit_multiple_tpu_devices(tmp_path):
    trainer_options = {
        "default_root_dir": tmp_path,
        "precision": "16-true",
        "enable_progress_bar": False,
        "max_epochs": 1,
        "accelerator": "tpu",
        "devices": "auto",
        "limit_train_batches": 3,
        "limit_val_batches": 3,
    }

    # multiple cores needs a big dataset
    model = SerialLoaderBoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False, min_acc=0.05)


class CustomBoringModel(BoringModel):
    def validation_step(self, *args, **kwargs):
        out = super().validation_step(*args, **kwargs)
        self.log("val_loss", out["x"])
        return out


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_early_stop(tmp_path):
    model = CustomBoringModel()
    trainer = Trainer(
        callbacks=[EarlyStopping(monitor="val_loss")],
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        accelerator="tpu",
        devices="auto",
    )
    trainer.fit(model)
    trainer.test(model, dataloaders=DataLoader(RandomDataset(32, 2000), batch_size=32))


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_grad_norm(tmp_path):
    """Test if grad_norm works on TPU."""
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 4,
        "accelerator": "tpu",
        "devices": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.4,
        "gradient_clip_val": 0.5,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_clip_grad_by_value(tmp_path):
    """Test if clip_gradients by value works on TPU."""
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 4,
        "accelerator": "tpu",
        "devices": 1,
        "limit_train_batches": 3,
        "limit_val_batches": 3,
        "gradient_clip_val": 0.5,
        "gradient_clip_algorithm": "value",
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_dataloaders_passed_to_fit(tmp_path):
    """Test if dataloaders passed to trainer works on TPU."""
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="tpu", devices="auto")
    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())


@pytest.mark.parametrize("devices", [[1, 8], "9, ", [9], [-1], 2, 10])
def test_tpu_misconfiguration(devices, tpu_available):
    with pytest.raises(ValueError, match="`devices` can only be"):
        Trainer(accelerator="tpu", devices=devices)


@pytest.mark.skipif(XLAAccelerator.is_available(), reason="test requires missing TPU")
@mock.patch("lightning.fabric.accelerators.xla._using_pjrt", return_value=True)
def test_exception_when_no_tpu_found(_, xla_available):
    """Test if exception is thrown when xla devices are not available."""
    with pytest.raises(MisconfigurationException, match="XLAAccelerator` can not run on your system"):
        Trainer(accelerator="tpu", devices=8)


@pytest.mark.parametrize("devices", [1, 4, [1]])
@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_accelerator_set_when_using_tpu(devices):
    """Test if the accelerator is set to `tpu` when devices is not None."""
    assert isinstance(Trainer(accelerator="tpu", devices=devices).accelerator, XLAAccelerator)


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_if_test_works_with_checkpoint_false(tmp_path):
    """Ensure that model trains properly when `enable_checkpointing` is set to False."""
    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(
        max_epochs=1,
        accelerator="tpu",
        devices="auto",
        default_root_dir=tmp_path,
        fast_dev_run=True,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


def wrap_launch_function(fn, strategy, *args, **kwargs):
    # the launcher does not manage this automatically. explanation available in:
    # https://github.com/Lightning-AI/lightning/pull/14926#discussion_r982976718
    strategy.setup_environment()
    return fn(*args, **kwargs)


def xla_launch(fn):
    # TODO: the accelerator should be optional to just launch processes, but this requires lazy initialization
    accelerator = XLAAccelerator()
    strategy = XLAStrategy(
        accelerator=accelerator,
        parallel_devices=XLAAccelerator.get_parallel_devices(XLAAccelerator.auto_device_count()),
    )
    launcher = _XLALauncher(strategy=strategy)
    wrapped = partial(wrap_launch_function, fn, strategy)
    return launcher.launch(wrapped, strategy)


def tpu_sync_dist_fn(strategy):
    sync = _Sync(strategy.reduce, _should=True, _op=torch.distributed.ReduceOp.SUM)
    value = torch.tensor([1.0])
    value = sync(value)
    assert value.item() == strategy.world_size


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_sync_dist():
    """Test tpu spawn sync dist operation."""
    xla_launch(tpu_sync_dist_fn)


class AssertXLADebugModel(BoringModel):
    def on_train_start(self):
        assert os.environ.get("PT_XLA_DEBUG") == "1", "PT_XLA_DEBUG was not set in environment variables"

    def teardown(self, stage):
        assert "PT_XLA_DEBUG" not in os.environ


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_debug_mode(tmp_path):
    """Test if debug mode works on TPU."""
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 4,
        "accelerator": "tpu",
        "devices": "auto",
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.4,
        "strategy": XLAStrategy(debug=True),
    }

    model = AssertXLADebugModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)


@RunIf(tpu=True)
def test_device_type_when_tpu_strategy_passed(tmp_path):
    trainer = Trainer(default_root_dir=tmp_path, strategy=XLAStrategy(), accelerator="tpu", devices="auto")
    assert isinstance(trainer.strategy, XLAStrategy)
    assert isinstance(trainer.accelerator, XLAAccelerator)
