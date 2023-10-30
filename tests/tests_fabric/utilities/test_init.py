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
import time
from unittest import mock
from unittest.mock import Mock

import pytest
import torch.nn
from lightning.fabric.utilities.init import _EmptyInit

from tests_fabric.helpers.runif import RunIf


@mock.patch("lightning.fabric.utilities.init._TORCH_GREATER_EQUAL_1_13", False)
def test_module_init_context_empty_init_support():
    with pytest.raises(NotImplementedError, match="Emtpy weight initialization requires PyTorch >= 1.13"), _EmptyInit():
        pass


@RunIf(min_cuda_gpus=1, min_torch="1.13")
def test_empty_init(monkeypatch):
    """Test that `_EmptyInit()` skips initialization and allocates uninitialized memory."""
    init_mock = Mock()
    monkeypatch.setattr(torch.Tensor, "uniform_", init_mock)

    with _EmptyInit(enabled=True):
        torch.nn.Linear(2, 2, device="cuda")
    init_mock.assert_not_called()

    with _EmptyInit(enabled=False):
        torch.nn.Linear(2, 2, device="cuda")
    init_mock.assert_called()


@RunIf(min_cuda_gpus=1, min_torch="1.13")
def test_empty_init_speed():
    """Test that `_EmptyInit()` is faster than regular initialization."""
    t0 = time.perf_counter()
    with _EmptyInit(enabled=False):
        torch.nn.Linear(10000, 10000, device="cuda")
    torch.cuda.synchronize()
    normal_init_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    with _EmptyInit(enabled=True):
        torch.nn.Linear(10000, 10000, device="cuda")
    torch.cuda.synchronize()
    empty_init_time = time.perf_counter() - t0

    assert normal_init_time > 2 * empty_init_time
