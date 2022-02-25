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
from unittest import mock

from pytorch_lightning.accelerators import CPUAccelerator, GPUAccelerator, IPUAccelerator, TPUAccelerator


@mock.patch("torch.cuda.device_count", return_value=2)
def test_auto_device_count(device_count_mock):
    assert CPUAccelerator.auto_device_count() == 1
    assert GPUAccelerator.auto_device_count() == 2
    assert TPUAccelerator.auto_device_count() == 8
    assert IPUAccelerator.auto_device_count() == 4
