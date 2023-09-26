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

import pytest
import torch
from lightning.fabric.accelerators.cpu import CPUAccelerator, _parse_cpu_cores


def test_auto_device_count():
    assert CPUAccelerator.auto_device_count() == 1


def test_availability():
    assert CPUAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Device should be CPU"):
        CPUAccelerator().setup_device(torch.device("cuda"))


@pytest.mark.parametrize(
    ("devices", "expected"),
    [(1, [torch.device("cpu")]), (2, [torch.device("cpu")] * 2), ("3", [torch.device("cpu")] * 3)],
)
def test_get_parallel_devices(devices, expected):
    assert CPUAccelerator.get_parallel_devices(devices) == expected


@pytest.mark.parametrize("devices", [[3], -1])
def test_invalid_devices_with_cpu_accelerator(devices):
    """Test invalid device flag raises MisconfigurationException."""
    with pytest.raises(TypeError, match="should be an int > 0"):
        _parse_cpu_cores(devices)
