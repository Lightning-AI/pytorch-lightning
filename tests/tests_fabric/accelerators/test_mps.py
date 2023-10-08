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
from lightning.fabric.accelerators.mps import MPSAccelerator
from lightning.fabric.utilities.exceptions import MisconfigurationException

from tests_fabric.helpers.runif import RunIf

_MAYBE_MPS = "mps" if MPSAccelerator.is_available() else "cpu"


def test_auto_device_count():
    assert MPSAccelerator.auto_device_count() == 1


@RunIf(mps=True)
def test_mps_availability():
    assert MPSAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Device should be MPS"):
        MPSAccelerator().setup_device(torch.device("cpu"))


@RunIf(mps=True)
@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        (1, [torch.device(_MAYBE_MPS, 0)]),
        ([0], [torch.device(_MAYBE_MPS, 0)]),
        ("1", [torch.device(_MAYBE_MPS, 0)]),
        ("0,", [torch.device(_MAYBE_MPS, 0)]),
    ],
)
def test_get_parallel_devices(devices, expected):
    assert MPSAccelerator.get_parallel_devices(devices) == expected


@RunIf(mps=True)
@pytest.mark.parametrize("devices", [2, [0, 2], "2", "0,2"])
def test_get_parallel_devices_invalid_request(devices):
    with pytest.raises(MisconfigurationException, match="But your machine only has"):
        MPSAccelerator.get_parallel_devices(devices)
