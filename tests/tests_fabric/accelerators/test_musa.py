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

from lightning.fabric.accelerators.musa import MUSAAccelerator
from tests_fabric.helpers.runif import RunIf

_MAYBE_MUSA = "musa" if MUSAAccelerator.is_available() else "cpu"


@mock.patch("lightning.fabric.accelerators.musa.num_musa_devices", return_value=2)
@RunIf(musa=True)
def test_auto_device_count(_):
    assert MUSAAccelerator.auto_device_count() == 2


@RunIf(musa=True)
def test_musa_availability():
    assert MUSAAccelerator.is_available()


def test_init_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Device should be MUSA"):
        MUSAAccelerator().setup_device(torch.device("cpu"))


@RunIf(musa=True)
@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        ([], []),
        ([1], [torch.device(_MAYBE_MUSA, 1)]),
        ([3, 1], [torch.device(_MAYBE_MUSA, 3), torch.device(_MAYBE_MUSA, 1)]),
    ],
)
def test_get_parallel_devices(devices, expected):
    assert MUSAAccelerator.get_parallel_devices(devices) == expected


@mock.patch("torch.musa.set_device")
@mock.patch("torch.musa.get_device_capability", return_value=(7, 0))
def test_set_cuda_device(_, set_device_mock):
    device = torch.device(_MAYBE_MUSA, 1)
    MUSAAccelerator().setup_device(device)
    set_device_mock.assert_called_once_with(device)
