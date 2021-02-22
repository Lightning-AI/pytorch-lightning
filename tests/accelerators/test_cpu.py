from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins import SingleDevicePlugin
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin, PrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException





def test_invalid_root_device():
    """ Test that CPU Accelerator has root device on CPU. """
    trainer = Mock()
    model = Mock()
    accelerator = CPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cuda", 1)),
        precision_plugin=PrecisionPlugin()
    )
    with pytest.raises(MisconfigurationException, match="Device should be CPU"):
        accelerator.setup(trainer=trainer, model=model)


def test_unsupported_precision_plugins():
    """ Test error messages are raised for unsupported precision plugins with CPU. """
    trainer = Mock()
    model = Mock()
    accelerator = CPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cpu")),
        precision_plugin=MixedPrecisionPlugin()
    )
    with pytest.raises(MisconfigurationException, match=r"amp \+ cpu is not supported."):
        accelerator.setup(trainer=trainer, model=model)
