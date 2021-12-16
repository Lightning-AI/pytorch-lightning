import os
from pathlib import Path
from typing import Any, Dict, Union

import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins import SingleDevicePlugin
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from tests.helpers.boring_model import BoringModel


def test_restore_checkpoint_after_pre_dispatch_default():
    """Assert default for restore_checkpoint_after_pre_dispatch is False."""
    plugin = SingleDevicePlugin(
        accelerator=CPUAccelerator(), device=torch.device("cpu"), precision_plugin=PrecisionPlugin()
    )
    assert not plugin.restore_checkpoint_after_pre_dispatch


@pytest.mark.parametrize("restore_after_pre_dispatch", [True, False])
def test_restore_checkpoint_after_pre_dispatch(tmpdir, restore_after_pre_dispatch):
    """Test to ensure that if restore_checkpoint_after_pre_dispatch is True, then we only load the state after pre-
    dispatch is called."""

    class TestPlugin(SingleDevicePlugin):
        predispatched_called = False

        def pre_dispatch(self, trainer: "pl.Trainer") -> None:
            super().pre_dispatch(trainer)
            self.predispatched_called = True

        @property
        def restore_checkpoint_after_pre_dispatch(self) -> bool:
            return restore_after_pre_dispatch

        def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
            assert self.predispatched_called == restore_after_pre_dispatch
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
    assert plugin.restore_checkpoint_after_pre_dispatch == restore_after_pre_dispatch

    trainer = Trainer(default_root_dir=tmpdir, strategy=plugin, fast_dev_run=True)
    trainer.fit(model, ckpt_path=checkpoint_path)
    for func in (trainer.test, trainer.validate, trainer.predict):
        plugin.predispatched_called = False
        func(model, ckpt_path=checkpoint_path)
