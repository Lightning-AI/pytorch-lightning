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
from torch.utils.data import DataLoader

import tests_pytorch.helpers.pipelines as tpipes
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import TPUAccelerator
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.strategies import XLAStrategy
from lightning.pytorch.strategies.launchers.xla import _XLALauncher
from lightning.pytorch.trainer.connectors.logger_connector.result import _Sync
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf


class SerialLoaderBoringModel(BoringModel):
    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 2000), batch_size=32)

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 2000), batch_size=32)


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_devices_1(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = {
        "default_root_dir": tmpdir,
        "enable_progress_bar": False,
        "max_epochs": 2,
        "accelerator": "tpu",
        "devices": 1,
        "limit_train_batches": 4,
        "limit_val_batches": 4,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)


@pytest.mark.parametrize("tpu_core", [1, 5])
@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_index(tmpdir, tpu_core):
    """Make sure model trains on TPU."""
    trainer_options = {
        "default_root_dir": tmpdir,
        "enable_progress_bar": False,
        "max_epochs": 2,
        "accelerator": "tpu",
        "devices": [tpu_core],
        "limit_train_batches": 4,
        "limit_val_batches": 4,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)
    import torch_xla

    assert torch_xla._XLAC._xla_get_default_device() == f"xla:{tpu_core}"


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_devices_8(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = {
        "default_root_dir": tmpdir,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "accelerator": "tpu",
        "devices": 8,
        "limit_train_batches": 4,
        "limit_val_batches": 4,
    }

    # 8 cores needs a big dataset
    model = SerialLoaderBoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False, min_acc=0.05)


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_16bit_tpu_devices_1(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = {
        "default_root_dir": tmpdir,
        "precision": "16-mixed",
        "enable_progress_bar": False,
        "max_epochs": 2,
        "accelerator": "tpu",
        "devices": 1,
        "limit_train_batches": 8,
        "limit_val_batches": 2,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.parametrize("tpu_core", [1, 5])
@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_16bit_tpu_index(tmpdir, tpu_core):
    """Make sure model trains on TPU."""
    trainer_options = {
        "default_root_dir": tmpdir,
        "precision": "16-mixed",
        "enable_progress_bar": False,
        "max_epochs": 2,
        "accelerator": "tpu",
        "devices": [tpu_core],
        "limit_train_batches": 4,
        "limit_val_batches": 2,
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)
    import torch_xla

    assert torch_xla._XLAC._xla_get_default_device() == f"xla:{tpu_core}"


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_16bit_tpu_devices_8(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = {
        "default_root_dir": tmpdir,
        "precision": "16-mixed",
        "enable_progress_bar": False,
        "max_epochs": 1,
        "accelerator": "tpu",
        "devices": 8,
        "limit_train_batches": 4,
        "limit_val_batches": 4,
    }

    # 8 cores needs a big dataset
    model = SerialLoaderBoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False, min_acc=0.05)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_model_tpu_early_stop(tmpdir):
    """Test if single TPU core training works."""

    class CustomBoringModel(BoringModel):
        def validation_step(self, *args, **kwargs):
            out = super().validation_step(*args, **kwargs)
            self.log("val_loss", out["x"])
            return out

    model = CustomBoringModel()
    trainer = Trainer(
        callbacks=[EarlyStopping(monitor="val_loss")],
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        accelerator="tpu",
        devices=8,
    )
    trainer.fit(model)
    trainer.test(dataloaders=DataLoader(RandomDataset(32, 2000), batch_size=32))


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_grad_norm(tmpdir):
    """Test if grad_norm works on TPU."""
    trainer_options = {
        "default_root_dir": tmpdir,
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


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_clip_grad_by_value(tmpdir):
    """Test if clip_gradients by value works on TPU."""
    trainer_options = {
        "default_root_dir": tmpdir,
        "enable_progress_bar": False,
        "max_epochs": 4,
        "accelerator": "tpu",
        "devices": 1,
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "gradient_clip_val": 0.5,
        "gradient_clip_algorithm": "value",
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_dataloaders_passed_to_fit(tmpdir):
    """Test if dataloaders passed to trainer works on TPU."""
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator="tpu", devices=8)
    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())


@pytest.mark.parametrize("devices", [[1, 8], "9, ", [9], [0], 2, 10])
def test_tpu_misconfiguration(devices, tpu_available):
    with pytest.raises(TypeError, match="`devices` can only be"):
        Trainer(accelerator="tpu", devices=devices)


@pytest.mark.skipif(TPUAccelerator.is_available(), reason="test requires missing TPU")
def test_exception_when_no_tpu_found(xla_available):
    """Test if exception is thrown when xla devices are not available."""
    with pytest.raises(MisconfigurationException, match="TPUAccelerator` can not run on your system"):
        Trainer(accelerator="tpu", devices=8)


@pytest.mark.parametrize("devices", [1, 8, [1]])
@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_accelerator_set_when_using_tpu(devices):
    """Test if the accelerator is set to `tpu` when devices is not None."""
    assert isinstance(Trainer(accelerator="tpu", devices=devices).accelerator, TPUAccelerator)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_if_test_works_with_checkpoint_false(tmpdir):
    """Ensure that model trains properly when `enable_checkpointing` is set to False."""
    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(
        max_epochs=1,
        accelerator="tpu",
        devices=8,
        default_root_dir=tmpdir,
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
    accelerator = TPUAccelerator()
    strategy = XLAStrategy(accelerator=accelerator, parallel_devices=list(range(8)))
    launcher = _XLALauncher(strategy=strategy)
    wrapped = partial(wrap_launch_function, fn, strategy)
    return launcher.launch(wrapped, strategy)


def tpu_sync_dist_fn(strategy):
    sync = _Sync(strategy.reduce, _should=True, _op=torch.distributed.ReduceOp.SUM)
    value = torch.tensor([1.0])
    value = sync(value)
    assert value.item() == 8


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_sync_dist():
    """Test tpu spawn sync dist operation."""
    xla_launch(tpu_sync_dist_fn)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_debug_mode(tmpdir):
    """Test if debug mode works on TPU."""

    class DebugModel(BoringModel):
        def on_train_start(self):
            assert os.environ.get("PT_XLA_DEBUG") == str(1), "PT_XLA_DEBUG was not set in environment variables"

        def teardown(self, stage):
            assert "PT_XLA_DEBUG" not in os.environ

    trainer_options = {
        "default_root_dir": tmpdir,
        "enable_progress_bar": False,
        "max_epochs": 4,
        "accelerator": "tpu",
        "devices": 8,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.4,
        "strategy": XLAStrategy(debug=True),
    }

    model = DebugModel()
    tpipes.run_model_test(trainer_options, model, with_hpc=False)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_host_world_size(tmpdir):
    """Test Host World size env setup on TPU."""

    class DebugModel(BoringModel):
        def on_train_start(self):
            assert os.environ.get("XRT_HOST_WORLD_SIZE") == str(1)

    trainer_options = {
        "default_root_dir": tmpdir,
        "enable_progress_bar": False,
        "max_epochs": 4,
        "accelerator": "tpu",
        "devices": 8,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.4,
    }

    model = DebugModel()
    assert "XRT_HOST_WORLD_SIZE" not in os.environ
    tpipes.run_model_test(trainer_options, model, with_hpc=False)
    assert "XRT_HOST_WORLD_SIZE" not in os.environ


@RunIf(tpu=True)
def test_device_type_when_tpu_strategy_passed(tmpdir):
    trainer = Trainer(default_root_dir=tmpdir, strategy=XLAStrategy(), accelerator="tpu", devices=8)
    assert isinstance(trainer.strategy, XLAStrategy)
    assert isinstance(trainer.accelerator, TPUAccelerator)
