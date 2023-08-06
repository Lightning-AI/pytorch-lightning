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

from lightning.fabric import Fabric
from tests_fabric.helpers.runif import RunIf


@pytest.mark.parametrize("strategy", ["ddp_spawn", pytest.param("ddp_fork", marks=RunIf(skip_windows=True))])
def test_memory_sharing_disabled(strategy):
    """Test that the multiprocessing launcher disables memory sharing on model parameters and buffers to avoid race
    conditions on model updates."""
    tensor = torch.rand(4)
    module = torch.nn.Linear(2, 2)
    assert not tensor.is_shared()
    assert not module.weight.is_shared()

    fabric = Fabric(accelerator="cpu", devices=2, strategy=strategy)
    fabric.launch(_test_memory_sharing_disabled, tensor, module=module)


def _test_memory_sharing_disabled(fabric, tensor, module):
    is_spawn = fabric.strategy.launcher._start_method == "spawn"
    assert not is_spawn or tensor.is_shared()
    assert not module.weight.is_shared()
