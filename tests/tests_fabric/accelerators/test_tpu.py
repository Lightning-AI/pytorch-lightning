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
# limitations under the License
import os

import pytest

from lightning.fabric.accelerators.tpu import TPUAccelerator
from tests_fabric.helpers.runif import RunIf


@RunIf(tpu=True)
def test_auto_device_count():
    assert TPUAccelerator.auto_device_count() == int(os.environ["TPU_NUM_DEVICES"])


@RunIf(tpu=True)
def test_availability():
    assert TPUAccelerator.is_available()


@pytest.mark.parametrize("devices", (1, 8))
def test_get_parallel_devices(devices, tpu_available):
    expected = TPUAccelerator.get_parallel_devices(devices)
    assert len(expected) == devices


def test_get_parallel_devices_raises(tpu_available):
    with pytest.raises(ValueError, match="devices` can only be"):
        TPUAccelerator.get_parallel_devices(0)
    with pytest.raises(ValueError, match="devices` can only be"):
        TPUAccelerator.get_parallel_devices(5)
    with pytest.raises(ValueError, match="Could not parse.*anything-else'"):
        TPUAccelerator.get_parallel_devices("anything-else")
