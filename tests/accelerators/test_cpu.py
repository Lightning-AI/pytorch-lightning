from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins import SingleDevicePlugin, NativeMixedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_unsupported_precision_plugins():
    """ Test error messages are raised for unsupported precision plugins with CPU. """
    trainer = Mock()
    model = Mock()
    accelerator = CPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cpu")),
        precision_plugin=NativeMixedPrecisionPlugin()
    )
    with pytest.raises(MisconfigurationException, match=r"amp \+ cpu is not supported."):
        accelerator.setup(trainer=trainer, model=model)
