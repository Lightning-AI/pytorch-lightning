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

import pytest
import torch
import torch.nn as nn
from lightning.fabric import Fabric

from tests_fabric.helpers.runif import RunIf


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.tied_layer = nn.Linear(2, 2)
        self.tied_layer.weight = self.layer.weight
        self.register_buffer("buffer", torch.ones(3))


@pytest.mark.parametrize("strategy", ["ddp_spawn", pytest.param("ddp_fork", marks=RunIf(skip_windows=True))])
def test_memory_sharing_disabled(strategy):
    """Test that the multiprocessing launcher disables memory sharing on model parameters and buffers to avoid race
    conditions on model updates."""
    tensor = torch.rand(4)
    model = SimpleModel()
    assert not tensor.is_shared()
    assert not model.layer.weight.is_shared()
    assert model.layer.weight.data_ptr() == model.tied_layer.weight.data_ptr()

    fabric = Fabric(accelerator="cpu", devices=2, strategy=strategy)
    fabric.launch(_test_memory_sharing_disabled, tensor, model)


def _test_memory_sharing_disabled(fabric, tensor, model):
    is_spawn = fabric.strategy.launcher._start_method == "spawn"
    assert not is_spawn or tensor.is_shared()
    assert not model.layer.weight.is_shared()
    assert not model.tied_layer.weight.is_shared()
    assert not model.buffer.is_shared()

    # weights remain tied
    assert model.layer.weight.data_ptr() == model.tied_layer.weight.data_ptr()
    assert torch.equal(model.layer.weight.data, model.tied_layer.weight.data)
    fabric.barrier()
