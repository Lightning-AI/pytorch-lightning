import os
from pathlib import Path
from typing import Any, Dict, Union
from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins import SingleDevicePlugin
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.debugging_examples import BoringModel
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_unsupported_precision_plugins():
    """Test error messages are raised for unsupported precision plugins with CPU."""
    trainer = Mock()
    accelerator = CPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cpu")), precision_plugin=MixedPrecisionPlugin()
    )
    with pytest.raises(MisconfigurationException, match=r"AMP \+ CPU is not supported"):
        accelerator.setup(trainer=trainer)


@pytest.mark.parametrize("delay_dispatch", [True, False])
def test_plugin_setup_optimizers_in_pre_dispatch(tmpdir, delay_dispatch):
    """
    Test when using a custom training type plugin that delays setup optimizers,
    we do not call setup optimizers till ``pre_dispatch``.
    """

    class TestModel(BoringModel):
        def on_fit_start(self):
            if delay_dispatch:
                # Ensure we haven't setup optimizers if we've delayed dispatch
                assert len(self.trainer.optimizers) == 0
            else:
                assert len(self.trainer.optimizers) > 0

        def on_fit_end(self):
            assert len(self.trainer.optimizers) > 0

    class CustomPlugin(SingleDevicePlugin):
        @property
        def setup_optimizers_in_pre_dispatch(self) -> bool:
            return delay_dispatch

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, plugins=CustomPlugin(device=torch.device("cpu")))
    trainer.fit(model)


def test_restore_checkpoint_after_pre_dispatch_default():
    """
    Assert default for restore_checkpoint_after_pre_dispatch is False.
    """
    plugin = SingleDevicePlugin(torch.device("cpu"))
    accelerator = CPUAccelerator(training_type_plugin=plugin, precision_plugin=PrecisionPlugin())
    assert not accelerator.restore_checkpoint_after_pre_dispatch
    assert not plugin.restore_checkpoint_after_pre_dispatch


@pytest.mark.parametrize("restore_after_pre_dispatch", [True, False])
def test_restore_checkpoint_after_pre_dispatch(tmpdir, restore_after_pre_dispatch):
    """
    Test to ensure that if restore_checkpoint_after_pre_dispatch is True, then we only load the state after
    pre-dispatch is called.
    """

    class TestPlugin(SingleDevicePlugin):
        predispatched_called = False

        def pre_dispatch(self) -> None:
            super().pre_dispatch()
            self.predispatched_called = True

        @property
        def restore_checkpoint_after_pre_dispatch(self) -> bool:
            return restore_after_pre_dispatch

        def load_checkpoint_file(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
            assert self.predispatched_called == restore_after_pre_dispatch
            return super().load_checkpoint_file(checkpoint_path)

    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    plugin = TestPlugin(torch.device("cpu"), checkpoint_io=TorchCheckpointIO())
    accelerator = CPUAccelerator(training_type_plugin=plugin, precision_plugin=PrecisionPlugin())

    assert accelerator.restore_checkpoint_after_pre_dispatch == restore_after_pre_dispatch
    assert plugin.restore_checkpoint_after_pre_dispatch == restore_after_pre_dispatch

    trainer = Trainer(
        default_root_dir=tmpdir, accelerator=accelerator, fast_dev_run=True, resume_from_checkpoint=checkpoint_path
    )
    trainer.fit(model)
    for func in (trainer.test, trainer.validate, trainer.predict):
        accelerator.training_type_plugin.predispatched_called = False
        func(model, ckpt_path=checkpoint_path)
