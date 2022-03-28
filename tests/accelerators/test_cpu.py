import os
from pathlib import Path
from typing import Any, Dict, Union

import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel


def test_restore_checkpoint_after_pre_setup_default():
    """Assert default for restore_checkpoint_after_setup is False."""
    plugin = SingleDeviceStrategy(
        accelerator=CPUAccelerator(), device=torch.device("cpu"), precision_plugin=PrecisionPlugin()
    )
    assert not plugin.restore_checkpoint_after_setup


def test_availability():
    assert CPUAccelerator.is_available()


@pytest.mark.parametrize("restore_after_pre_setup", [True, False])
def test_restore_checkpoint_after_pre_setup(tmpdir, restore_after_pre_setup):
    """Test to ensure that if restore_checkpoint_after_setup is True, then we only load the state after pre-
    dispatch is called."""

    class TestPlugin(SingleDeviceStrategy):
        setup_called = False

        def setup(self, trainer: "pl.Trainer") -> None:
            super().setup(trainer)
            self.setup_called = True

        @property
        def restore_checkpoint_after_setup(self) -> bool:
            return restore_after_pre_setup

        def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
            assert self.setup_called == restore_after_pre_setup
            return super().load_checkpoint(checkpoint_path)

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    plugin = TestPlugin(
        accelerator=CPUAccelerator(),
        precision_plugin=PrecisionPlugin(),
        device=torch.device("cpu"),
        checkpoint_io=TorchCheckpointIO(),
    )
    assert plugin.restore_checkpoint_after_setup == restore_after_pre_setup

    trainer = Trainer(default_root_dir=tmpdir, strategy=plugin, fast_dev_run=True)
    trainer.fit(model, ckpt_path=checkpoint_path)
    for func in (trainer.test, trainer.validate, trainer.predict):
        plugin.setup_called = False
        func(model, ckpt_path=checkpoint_path)


@pytest.mark.parametrize("devices", ([3], -1, 0))
def test_invalid_devices_with_cpu_accelerator(devices):
    """Test invalid device flag raises MisconfigurationException with CPUAccelerator."""
    with pytest.raises(MisconfigurationException, match="should be an int > 0"):
        Trainer(accelerator="cpu", devices=devices)
