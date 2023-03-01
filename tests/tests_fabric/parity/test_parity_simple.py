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
from copy import deepcopy
from typing import Callable

import pytest
import torch
import torch.distributed
import torch.nn.functional
from tests_fabric.helpers.runif import RunIf
from tests_fabric.parity.models import ConvNet
from tests_fabric.parity.utils import is_state_dict_equal, make_deterministic

from lightning.fabric.fabric import Fabric
from tests_fabric.parity.utils import get_model_input_dtype

NUM_STEPS_DEFAULT = 1000


def train_torch(
    move_to_device: Callable,
    precision_context,
    input_dtype=torch.float32,
    num_steps=NUM_STEPS_DEFAULT,
    batch_size=4,
):
    make_deterministic()
    model = ConvNet()
    model = move_to_device(model)
    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
    optimizer = model.get_optimizer()
    loss_fn = model.get_loss_function()

    model.train()
    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(num_steps):
        t0 = time.perf_counter()

        inputs, labels = next(iterator)
        inputs, labels = move_to_device(inputs), move_to_device(labels)
        optimizer.zero_grad()
        with precision_context():
            outputs = model(inputs.to(input_dtype))
        loss = loss_fn(outputs.float(), labels)
        loss.backward()
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    return model.state_dict(), torch.tensor(iteration_timings)


def train_fabric(fabric, num_steps=NUM_STEPS_DEFAULT, batch_size=4):
    make_deterministic()

    model = ConvNet()
    initial_state_dict = deepcopy(model.state_dict())

    optimizer = model.get_optimizer()
    model, optimizer = fabric.setup(model, optimizer)

    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
    dataloader = fabric.setup_dataloaders(dataloader)
    loss_fn = model.get_loss_function()

    model.train()
    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(num_steps):
        t0 = time.perf_counter()

        inputs, labels = next(iterator)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    # check that the model has changed
    assert not is_state_dict_equal(initial_state_dict, model.state_dict())

    return model.state_dict(), torch.tensor(iteration_timings)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "precision, accelerator",
    [
        (32, "cpu"),
        pytest.param(32, "gpu", marks=RunIf(min_cuda_gpus=1)),
        # pytest.param(16, "gpu", marks=RunIf(min_cuda_gpus=1)),  # TODO: requires GradScaler
        pytest.param("bf16", "cpu"),
        pytest.param("bf16", "gpu", marks=RunIf(min_cuda_gpus=1, bf16_cuda=True)),
        pytest.param(32, "mps", marks=RunIf(mps=True)),
    ],
)
def test_parity_single_device(precision, accelerator):
    input_dtype = get_model_input_dtype(precision)

    # Train with Fabric
    fabric = Fabric(precision=precision, accelerator=accelerator, devices=1)
    fabric_state_dict, timings_fabric = train_fabric(fabric)

    # Train with raw PyTorch
    torch_state_dict, timings_torch = train_torch(
        fabric.to_device, precision_context=fabric.autocast, input_dtype=input_dtype
    )

    # Compare the final weights
    assert is_state_dict_equal(torch_state_dict, fabric_state_dict)

    # Compare the time per iteration
    # Drop measurements of the first iterations, as they may be slower than others
    # The median is more robust to outliers than the mean
    assert torch.isclose(torch.median(timings_torch[3:]), torch.median(timings_fabric[3:]), rtol=1e-4, atol=1e-4)
