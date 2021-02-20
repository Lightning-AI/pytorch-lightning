from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.plugins import SingleDevicePlugin, PrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_invalid_root_device():
    trainer = Mock()
    model = Mock()
    accelerator = CPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cuda", 1)),
        precision_plugin=PrecisionPlugin()
    )
    with pytest.raises(MisconfigurationException, match="Device should be CPU"):
        accelerator.setup(trainer=trainer, model=model)
