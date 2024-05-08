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
from unittest.mock import Mock

import pytest
import torch.nn
from lightning.fabric.utilities.init import (
    _EmptyInit,
    _has_meta_device_parameters_or_buffers,
    _materialize_meta_tensors,
)

from tests_fabric.helpers.runif import RunIf


@RunIf(min_cuda_gpus=1)
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


@RunIf(min_cuda_gpus=1)
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


@RunIf(min_torch="2.1")
def test_materialize_meta_tensors():
    class Submodule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(1, 1)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buf", torch.tensor(0))
            self.l = torch.nn.Linear(1, 1)
            self.inner = Submodule()

    with torch.device("meta"):
        model = MyModel()

    with pytest.raises(TypeError, match="MyModel.reset_parameters` method is implemented"):
        _materialize_meta_tensors(model, torch.device("cpu"))

    class MyModel2(MyModel):
        def reset_parameters(self):
            self.buf = torch.empty_like(self.buf)

    with torch.device("meta"):
        model = MyModel2()

    _materialize_meta_tensors(model, torch.device("cpu"))
    assert model.buf.device.type == "cpu"
    assert len(list(model.parameters())) == 4
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_has_meta_device_parameters_or_buffers():
    """Test that the `_has_meta_device_parameters_or_buffers` function can find meta-device parameters in models and
    optimizers."""

    class BufferModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.ones(2, device="meta"))

    # nn.Module
    module = torch.nn.Linear(2, 2)
    meta_module = torch.nn.Linear(2, 2, device="meta")
    buffer_meta_module = BufferModule()
    assert not _has_meta_device_parameters_or_buffers(module)
    assert _has_meta_device_parameters_or_buffers(meta_module)
    assert _has_meta_device_parameters_or_buffers(torch.nn.Sequential(module, meta_module, torch.nn.ReLU()))
    assert _has_meta_device_parameters_or_buffers(buffer_meta_module)
    # optim.Optimizer
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    meta_optimizer = torch.optim.SGD(meta_module.parameters(), lr=0.1)
    assert not _has_meta_device_parameters_or_buffers(optimizer)
    assert _has_meta_device_parameters_or_buffers(meta_optimizer)
    # unsupported objects
    with pytest.raises(TypeError, match="Expected `torch.nn.Module` or `torch.optim.Optimizer`"):
        _has_meta_device_parameters_or_buffers(None)
