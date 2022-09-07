# Copyright The PyTorch Lightning team.
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
from unittest.mock import Mock

import pytest
import torch
from tests_lite.helpers.runif import RunIf

from lightning_lite.accelerators.cpu import CPUAccelerator


def test_auto_device_count():
    assert CPUAccelerator.auto_device_count() == 1


def test_availability():
    assert CPUAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Device should be CPU"):
        CPUAccelerator().init_device(torch.device("cuda"))


@pytest.mark.parametrize(
    "devices,expected",
    [
        (1, [torch.device("cpu")]),
        (2, [torch.device("cpu")] * 2),
        ("3", [torch.device("cpu")] * 3),
    ],
)
def test_get_parallel_devices(devices, expected):
    assert CPUAccelerator.get_parallel_devices(devices) == expected


@RunIf(psutil=True)
def test_get_device_stats(tmpdir):
    gpu_stats = CPUAccelerator().get_device_stats(Mock())
    fields = ["cpu_vm_percent", "cpu_percent", "cpu_swap_percent"]

    for f in fields:
        assert any(f in h for h in gpu_stats.keys())
