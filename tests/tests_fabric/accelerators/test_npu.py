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
from unittest import mock

import pytest
import torch

from lightning.fabric import Fabric
from lightning.fabric.accelerators import ACCELERATOR_REGISTRY
from lightning.fabric.accelerators.npu import NPUAccelerator, _parse_npu_devices, find_usable_npu_devices
from lightning.fabric.utilities.device_parser import _select_auto_accelerator
from tests_fabric.helpers.runif import RunIf


@mock.patch("lightning.fabric.accelerators.npu.num_npu_devices", return_value=2)
def test_auto_device_count(_):
    assert NPUAccelerator.auto_device_count() == 2


@RunIf(min_npu_gpus=1)
def test_npu_availability():
    assert NPUAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Device should be NPU"):
        NPUAccelerator().setup_device(torch.device("cpu"))


@RunIf(min_npu_gpus=1)
@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        ([], []),
        ([1], [torch.device("npu", 1)]),
        ([3, 1], [torch.device("npu", 3), torch.device("npu", 1)]),
    ],
)
def test_get_parallel_devices(devices, expected):
    assert NPUAccelerator.get_parallel_devices(devices) == expected


@mock.patch("torch.npu.set_device")
@RunIf(min_npu_gpus=1)
def test_set_npu_device(set_device_mock):
    device = torch.device("npu", 1)
    NPUAccelerator().setup_device(device)
    set_device_mock.assert_called_once_with(device)


@mock.patch("lightning.fabric.accelerators.npu.num_npu_devices", return_value=8)
def test_parse_npu_devices(_):
    assert _parse_npu_devices(None) is None
    assert _parse_npu_devices(0) is None
    assert _parse_npu_devices("0") is None
    assert _parse_npu_devices("[]") is None
    assert _parse_npu_devices(-1) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert _parse_npu_devices(2) == [0, 1]
    assert _parse_npu_devices("2,3") == [2, 3]
    assert _parse_npu_devices([1, 2]) == [1, 2]


@mock.patch("lightning.fabric.accelerators.npu.num_npu_devices", return_value=2)
def test_parse_npu_devices_out_of_range(_):
    with pytest.raises(ValueError, match="You requested npu"):
        _parse_npu_devices([3])


def test_npu_accelerator_in_registry():
    assert "npu" in ACCELERATOR_REGISTRY
    assert ACCELERATOR_REGISTRY["npu"]["accelerator"] is NPUAccelerator


@RunIf(min_npu_gpus=1)
def test_find_usable_npu_devices():
    usable = find_usable_npu_devices(1)
    assert isinstance(usable, list) and len(usable) == 1


@mock.patch("lightning.fabric.accelerators.xla.XLAAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.cuda.CUDAAccelerator.is_available", return_value=True)
@mock.patch("lightning.fabric.accelerators.npu.NPUAccelerator.is_available", return_value=True)
def test_auto_probe_prefers_npu_over_cuda(*_):
    assert _select_auto_accelerator() == "npu"


@mock.patch("lightning.fabric.accelerators.xla.XLAAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.cuda.CUDAAccelerator.is_available", return_value=True)
@mock.patch("lightning.fabric.accelerators.npu.NPUAccelerator.is_available", return_value=False)
def test_auto_probe_falls_back_to_cuda(*_):
    assert _select_auto_accelerator() == "cuda"


@RunIf(min_npu_gpus=1)
def test_fabric_npu_smoke():
    fabric = Fabric(accelerator="npu", devices=1)
    assert isinstance(fabric.accelerator, NPUAccelerator)
