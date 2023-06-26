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
def test_single_device_grad_clipping(clip_type, precision):
    fabric = Fabric(accelerator="auto", devices=1, precision=precision)
    _run_grad_clipping_test(fabric=fabric, clip_type=clip_type)


def _run_grad_clipping_test(fabric, clip_type):
    model = torch.nn.Linear(32, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    # 10 retries
    i = 0
    while True:
        batch = torch.rand(2, 32, device=fabric.device)
        loss = model(batch).sum()
        fabric.backward(loss)

        try:
            if clip_type == "norm":
                fabric.clip_gradients(model, optimizer, max_norm=0.05, error_if_nonfinite=True)
                parameters = model.parameters()
                grad_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [torch.linalg.vector_norm(p.grad.detach(), 2, dtype=torch.float32) for p in parameters]),
                    2,
                )
                torch.testing.assert_close(grad_norm, torch.tensor(0.05, device=self.device))

            else:
                for p in model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item():
                        raise RuntimeError("Nonfinite grads")
                fabric.clip_gradients(model, optimizer, clip_val=1e-10)
                parameters = model.parameters()
                grad_max_list = [torch.max(p.grad.detach().abs()) for p in parameters]
                grad_max = torch.max(torch.stack(grad_max_list))
                torch.testing.assert_close(grad_max.abs(), torch.tensor(1e-10, device=fabric.device))

            optimizer.step()
            optimizer.zero_grad()
            break

        except RuntimeError as ex:
            # nonfinite grads -> skip and continue
            # this may repeat until the scaler finds a factor where overflow is avoided,
            # so the while loop should eventually break
            # stop after a max of 10 tries
            if i > 10 or not str(ex).startswith("Nonfinite grads"):
                raise ex

            # unscale was already called by last attempt,
            # but no update afterwards since optimizer step was missing.
            # Manually update here -> Need to update inf stats first.
            scaler = getattr(fabric._precision, "scaler", None)
            if scaler is not None:
                scaler._check_inf_per_device(optimizer)
                scaler.update()
        finally:
            i += 1
