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
# limitations under the License
import pytest
from tests_lite.helpers.runif import RunIf

from lightning_lite.accelerators.tpu import TPUAccelerator


def test_auto_device_count():
    assert TPUAccelerator.auto_device_count() == 8


@RunIf(tpu=True)
def test_availability():
    assert TPUAccelerator.is_available()


@pytest.mark.parametrize(
    "devices,expected",
    [
        (0, []),  # TODO(lite): This should raise an exception
        (1, [0]),
        (2, [0, 1]),
        (3, [0, 1, 2]),
        ("anything-else", "anything-else"),  # TODO(lite): This should raise an exception
    ],
)
def test_get_parallel_devices(devices, expected):
    assert TPUAccelerator.get_parallel_devices(devices) == expected
