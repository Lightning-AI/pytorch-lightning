import logging
import os
from unittest import mock
from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins import PrecisionPlugin, SingleDevicePlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
def test_invalid_root_device():
    """ Test that GPU Accelerator has root device on GPU. """
    accelerator = GPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cpu")),
        precision_plugin=PrecisionPlugin()
    )
    with pytest.raises(MisconfigurationException, match="Device should be GPU"):
        accelerator.setup(trainer=Mock(), model=Mock())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multi-GPU machine")
def test_root_device_set():
    """ Test that GPU Accelerator sets the current device to the root device. """
    accelerator = GPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cuda", 1)),
        precision_plugin=PrecisionPlugin()
    )
    accelerator.setup(trainer=Mock(), model=Mock())
    assert torch.cuda.current_device() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@mock.patch.dict(os.environ, {"CUDA_DEVICE_ORDER": ""})
def test_cuda_environment_variables_set():
    """ Test that GPU Accelerator sets NVIDIA environment variables. """
    accelerator = GPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cuda", 0)),
        precision_plugin=PrecisionPlugin()
    )
    accelerator.setup(trainer=Mock(), model=Mock())
    assert os.getenv("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU machine")
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1, 2", "LOCAL_RANK": "3"})
def test_cuda_visible_devices_logged(caplog):
    """ Test that GPU Accelerator logs CUDA_VISIBLE_DEVICES env variable. """
    accelerator = GPUAccelerator(
        training_type_plugin=SingleDevicePlugin(torch.device("cuda", 0)),
        precision_plugin=PrecisionPlugin()
    )
    with caplog.at_level(logging.INFO):
        accelerator.setup(trainer=Mock(), model=Mock())
        assert "LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [1, 2]" in caplog.text
