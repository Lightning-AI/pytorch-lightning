from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins import SingleDevicePlugin
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel


def test_unsupported_precision_plugins():
    """ Test error messages are raised for unsupported precision plugins with CPU. """
    trainer = Mock()
    model = Mock()
    accelerator = CPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cpu")), precision_plugin=MixedPrecisionPlugin()
    )
    with pytest.raises(MisconfigurationException, match=r"AMP \+ CPU is not supported"):
        accelerator.setup(trainer=trainer, model=model)


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
