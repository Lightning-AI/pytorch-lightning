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

import pytest
import torch

from pytorch_lightning.utilities import device_parser


@pytest.mark.skipif(
    "fork" in torch.multiprocessing.get_all_start_methods(), reason="Requires platform without forking support"
)
@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.device_count", return_value=2)
def test_num_cuda_devices_without_forking(*_):
    """This merely tests that on platforms without fork support our helper functions fall back to the default
    implementation for determining cuda availability."""
    assert device_parser.is_cuda_available()
    assert device_parser.num_cuda_devices() == 2
