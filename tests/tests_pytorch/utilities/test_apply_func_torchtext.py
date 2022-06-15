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
import pytest
import torch

from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.imports import _TORCHTEXT_LEGACY
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.torchtext_utils import get_dummy_torchtext_data_iterator


@pytest.mark.parametrize("include_lengths", [False, True])
@pytest.mark.parametrize("device", [torch.device("cuda", 0)])
@pytest.mark.skipif(not _TORCHTEXT_LEGACY, reason="torchtext.legacy is deprecated.")
@RunIf(min_cuda_gpus=1)
def test_batch_move_data_to_device_torchtext_include_lengths(include_lengths, device):
    data_iterator, _ = get_dummy_torchtext_data_iterator(num_samples=3, batch_size=3, include_lengths=include_lengths)
    data_iter = iter(data_iterator)
    batch = next(data_iter)

    with pytest.deprecated_call(match="The `torchtext.legacy.Batch` object is deprecated"):
        batch_on_device = move_data_to_device(batch, device)

    if include_lengths:
        # tensor with data
        assert batch_on_device.text[0].device == device
        # tensor with length of data
        assert batch_on_device.text[1].device == device
    else:
        assert batch_on_device.text.device == device


@pytest.mark.parametrize("include_lengths", [False, True])
@pytest.mark.skipif(not _TORCHTEXT_LEGACY, reason="torchtext.legacy is deprecated.")
def test_batch_move_data_to_device_torchtext_include_lengths_cpu(include_lengths):
    test_batch_move_data_to_device_torchtext_include_lengths(include_lengths, torch.device("cpu"))
