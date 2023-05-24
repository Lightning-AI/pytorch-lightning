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

from lightning.fabric.plugins import FSDPPrecision
from tests_fabric.helpers.runif import RunIf


@mock.patch("lightning.fabric.plugins.precision.fsdp._TORCH_GREATER_EQUAL_1_12", False)
def test_fsdp_precision_support(*_):
    with pytest.raises(NotImplementedError, match="`FSDPPrecision` is supported from PyTorch v1.12.0"):
        FSDPPrecision(precision="16-mixed", device="cuda")


@RunIf(min_torch="1.12", min_cuda_gpus=1)
@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("16-mixed", (torch.float32, torch.float16, torch.float16)),
        ("bf16-mixed", (torch.float32, torch.bfloat16, torch.bfloat16)),
        ("16-true", (torch.float16, torch.float16, torch.float16)),
        ("bf16-true", (torch.bfloat16, torch.bfloat16, torch.bfloat16)),
    ],
)
def test_fsdp_precision_config(precision, expected):
    plugin = FSDPPrecision(precision=precision, device="cuda")
    config = plugin.mixed_precision_config

    assert config.param_dtype == expected[0]
    assert config.buffer_dtype == expected[1]
    assert config.reduce_dtype == expected[2]
