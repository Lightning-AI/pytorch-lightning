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
from tests_fabric.parity.models import TinyModel
from tests_fabric.parity.utils import (
    get_model_input_dtype,
    make_deterministic,
    TrackingMode,
)

from lightning.fabric.fabric import Fabric


def train_torch(
    move_to_device: Callable,
    precision_context,
    input_dtype=torch.float32,
):
    make_deterministic()

    model = TinyModel()
    model = move_to_device(model)
    dataloader = model.get_dataloader()
    optimizer = model.get_optimizer()
    loss_fn = model.get_loss_function()

    model.train()
    with TrackingMode() as tracked_calls:
        iterator = iter(dataloader)
        for _ in range(model.num_steps):
            inputs, labels = next(iterator)
            inputs, labels = move_to_device(inputs), move_to_device(labels)
            optimizer.zero_grad()
            with precision_context():
                outputs = model(inputs.to(input_dtype))
            loss = loss_fn(outputs.float(), labels)
            loss.backward()
            optimizer.step()

    return tracked_calls.calls


def train_fabric(fabric):
    make_deterministic()

    model = TinyModel()
    optimizer = model.get_optimizer()
    model, optimizer = fabric.setup(model, optimizer)

    dataloader = model.get_dataloader()
    dataloader = fabric.setup_dataloaders(dataloader)
    loss_fn = model.get_loss_function()

    model.train()
    with TrackingMode() as tracked_calls:
        iterator = iter(dataloader)
        for _ in range(model.num_steps):
            inputs, labels = next(iterator)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            fabric.backward(loss)
            optimizer.step()

    return tracked_calls.calls


@pytest.mark.usefixtures("reset_deterministic_algorithm", "reset_cudnn_benchmark")
@pytest.mark.parametrize(
    "precision, accelerator",
    [
        (32, "cpu"),
        # pytest.param(32, "cuda", marks=RunIf(min_cuda_gpus=1)),
        # pytest.param(16, "cuda", marks=RunIf(min_cuda_gpus=1)),  # TODO: requires GradScaler
        pytest.param("bf16", "cpu"),
        # pytest.param("bf16", "cuda", marks=RunIf(min_cuda_gpus=1, bf16_cuda=True)),
        pytest.param(32, "mps", marks=RunIf(mps=True)),
    ],
)
def test_parity_torch_calls(precision, accelerator):
    input_dtype = get_model_input_dtype(precision)

    # Train with Fabric
    fabric = Fabric(precision=precision, accelerator=accelerator, devices=1)
    calls_fabric = train_fabric(fabric)

    # Train with raw PyTorch
    calls_torch = train_torch(
        fabric.to_device, precision_context=fabric.autocast, input_dtype=input_dtype
    )

    # Compare the calls made to ATen
    assert calls_torch == calls_fabric
