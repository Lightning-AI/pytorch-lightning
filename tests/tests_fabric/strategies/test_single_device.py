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
from unittest.mock import Mock

import pytest
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import SingleDeviceStrategy

from tests_fabric.helpers.runif import RunIf


def test_single_device_default_device():
    assert SingleDeviceStrategy().root_device == torch.device("cpu")


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu"), "cuda:1", torch.device("cuda")])
def test_single_device_root_device(device):
    assert SingleDeviceStrategy(device).root_device == torch.device(device)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda", 3)])
def test_single_device_ranks(device):
    strategy = SingleDeviceStrategy(device)
    assert strategy.world_size == 1
    assert strategy.local_rank == 0
    assert strategy.global_rank == 0
    assert strategy.is_global_zero


def test_single_device_collectives():
    """Test that collectives in the single-device strategy act as the identity."""
    strategy = SingleDeviceStrategy()
    tensor = Mock()
    assert strategy.all_gather(tensor) == tensor
    assert strategy.all_reduce(tensor) == tensor
    assert strategy.broadcast(tensor) == tensor


def test_single_device_module_to_device():
    strategy = SingleDeviceStrategy()
    strategy._root_device = Mock()
    module = Mock(spec=torch.nn.Module)
    strategy.module_to_device(module)
    module.to.assert_called_with(strategy.root_device)


@pytest.mark.parametrize(
    "precision",
    [
        "32-true",
        pytest.param("16-mixed", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("bf16-mixed", marks=RunIf(bf16_cuda=True)),
    ],
)
@pytest.mark.parametrize("clip_type", ["norm", "val"])
def test_clip_gradients(clip_type, precision):
    if clip_type == "norm" and precision == "16-mixed":
        pytest.skip(reason="Clipping by norm with 16-mixed is numerically unstable.")

    fabric = Fabric(accelerator="auto", devices=1, precision=precision)
    _run_test_clip_gradients(fabric=fabric, clip_type=clip_type)


def _run_test_clip_gradients(fabric, clip_type):
    in_features, out_features = 32, 2
    model = torch.nn.Linear(in_features, out_features, bias=False)
    model.weight.data.fill_(0.01)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    batch = torch.full((1, in_features), 0.1, device=fabric.device)
    loss = model(batch).sum()

    # The example is constructed such that the gradients are all the same
    fabric.backward(loss)

    if clip_type == "norm":
        norm = torch.linalg.vector_norm(model.weight.grad.detach().cpu(), 2, dtype=torch.float32).item()
        new_norm = norm / 2.0
        fabric.clip_gradients(model, optimizer, max_norm=new_norm)
        assert torch.allclose(
            torch.linalg.vector_norm(model.weight.grad.detach().cpu(), 2, dtype=torch.float32),
            torch.tensor(new_norm),
        )
    elif clip_type == "val":
        val = model.weight.grad.view(-1)[0].item()
        new_val = val / 2.0
        fabric.clip_gradients(model, optimizer, clip_val=new_val)
        assert torch.allclose(model.weight.grad, torch.full_like(model.weight.grad, new_val))
    else:
        raise AssertionError(f"Unknown clip type: {clip_type}")

    optimizer.step()
    optimizer.zero_grad()
