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
"""Test deprecated functionality which will be removed in v1.8.0."""
from unittest.mock import Mock

import pytest
import torch

from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.enums import DeviceType, DistributedType
from pytorch_lightning.utilities.imports import _TORCHTEXT_LEGACY
from tests.helpers.torchtext_utils import get_dummy_torchtext_data_iterator


def test_v1_8_0_deprecated_distributed_type_enum():

    with pytest.deprecated_call(match="has been deprecated in v1.6 and will be removed in v1.8."):
        _ = DistributedType.DDP


def test_v1_8_0_deprecated_device_type_enum():

    with pytest.deprecated_call(match="has been deprecated in v1.6 and will be removed in v1.8."):
        _ = DeviceType.CPU


@pytest.mark.skipif(not _TORCHTEXT_LEGACY, reason="torchtext.legacy is deprecated.")
def test_v1_8_0_deprecated_torchtext_batch():

    with pytest.deprecated_call(match="is deprecated and Lightning will remove support for it in v1.8"):
        data_iterator, _ = get_dummy_torchtext_data_iterator(num_samples=3, batch_size=3)
        batch = next(iter(data_iterator))
        _ = move_data_to_device(batch=batch, device=torch.device("cpu"))


def test_v1_8_0_index_batch_sampler_wrapper_batch_indices():
    sampler = IndexBatchSamplerWrapper(Mock())
    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
        _ = sampler.batch_indices

    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8"):
        sampler.batch_indices = []
