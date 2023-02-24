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
from re import escape
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from tests_fabric.helpers.runif import RunIf

from lightning.fabric.accelerators.xpu import find_usable_xpu_devices, XPUAccelerator


@mock.patch("lightning.fabric.accelerators.xpu.num_xpu_devices", return_value=2)
def test_auto_device_count(_):
    assert XPUAccelerator.auto_device_count() == 2


@RunIf(min_xpu_gpus=1)
def test_gpu_availability():
    assert XPUAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Device should be XPU"):
        XPUAccelerator().setup_device(torch.device("cpu"))


@pytest.mark.parametrize(
    "devices,expected",
    [
        ([], []),
        ([1], [torch.device("xpu", 1)]),
        ([3, 1], [torch.device("xpu", 3), torch.device("xpu", 1)]),
    ],
)
def test_get_parallel_devices(devices, expected):
    assert XPUAccelerator.get_parallel_devices(devices) == expected


# @mock.patch("torch.xpu.set_device")
# @mock.patch("torch.xpu.get_device_capability", return_value=(7, 0))
# def test_set_xpu_device(_, set_device_mock):
#    device = torch.device("xpu", 1)
#    XPUAccelerator().setup_device(device)
#    set_device_mock.assert_called_once_with(device)


# @mock.patch("lightning.fabric.accelerators.xpu._device_count_nvml", return_value=-1)
# @mock.patch("torch.xpu.device_count", return_value=100)
# def test_num_xpu_devices_without_nvml(*_):
#    """Test that if NVML can't be loaded, our helper functions fall back to the default implementation for
#    determining XPU availability."""
#    num_xpu_devices.cache_clear()
#    assert is_xpu_available()
#    assert num_xpu_devices() == 100
#    num_xpu_devices.cache_clear()


# @mock.patch.dict(os.environ, {}, clear=True)
# def test_force_nvml_based_xpu_check():
#    """Test that we force PyTorch to use the NVML-based XPU checks."""
#    importlib.reload(lightning.fabric)  # reevaluate top-level code, without becoming a different object
#
#    assert os.environ["PYTORCH_NVML_BASED_XPU_CHECK"] == "1"


# @RunIf(min_torch="1.12")
# @mock.patch("torch.xpu.get_device_capability", return_value=(10, 1))
# @mock.patch("torch.xpu.get_device_name", return_value="Z100")
# def test_tf32_message(_, __, caplog, monkeypatch):
#
#    # for some reason, caplog doesn't work with our rank_zero_info utilities
#    monkeypatch.setattr(lightning.fabric.accelerators.xpu, "rank_zero_info", logging.info)
#
#    device = Mock()
#    expected = "Z100') that has Tensor Cores"
#    assert torch.get_float32_matmul_precision() == "highest"  # default in torch
#    with caplog.at_level(logging.INFO):
#        _check_xpu_matmul_precision(device)
#    assert expected in caplog.text
#
#    caplog.clear()
#    torch.backends.xpu.matmul.allow_tf32 = True  # changing this changes the string
#    assert torch.get_float32_matmul_precision() == "high"
#    with caplog.at_level(logging.INFO):
#        _check_xpu_math_precision(device)
#    assert not caplog.text
#
#    caplog.clear()
#    torch.backends.xpu.matmul.allow_tf32 = False
#    torch.set_float32_matmul_precision("medium")  # also the other way around
#    assert torch.backends.xpu.matmul.allow_tf32
#    with caplog.at_level(logging.INFO):
#        _check_xpu_math_precision(device)
#    assert not caplog.text
#
#    torch.set_float32_matmul_precision("highest")  # can be reverted
#    with caplog.at_level(logging.INFO):
#        _check_xpu_math_precision(device)
#    assert expected in caplog.text


def test_find_usable_xpu_devices_error_handling():
    """Test error handling for edge cases when using `find_usable_xpu_devices`."""

    # Asking for GPUs if no GPUs visible
    with mock.patch("lightning.fabric.accelerators.xpu.num_xpu_devices", return_value=0), pytest.raises(
        ValueError, match="You requested to find 2 devices but there are no visible XPU"
    ):
        find_usable_xpu_devices(2)

    # Asking for more GPUs than are visible
    with mock.patch("lightning.fabric.accelerators.xpu.num_xpu_devices", return_value=1), pytest.raises(
        ValueError, match="this machine only has 1 GPUs"
    ):
        find_usable_xpu_devices(2)

    # All GPUs are unusable
    tensor_mock = Mock(side_effect=RuntimeError)  # simulate device placement fails
    with mock.patch("lightning.fabric.accelerators.xpu.num_xpu_devices", return_value=2), mock.patch(
        "lightning.fabric.accelerators.xpu.torch.tensor", tensor_mock
    ), pytest.raises(RuntimeError, match=escape("The devices [0, 1] are occupied by other processes")):
        find_usable_xpu_devices(2)
