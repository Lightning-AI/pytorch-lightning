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
import os
import time
from copy import deepcopy

import pytest
import torch
import torch.distributed
import torch.nn.functional
from tests_fabric.helpers.runif import RunIf
from tests_fabric.parity.models import ConvNet
from tests_fabric.parity.utils import is_state_dict_equal, make_deterministic
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lightning.fabric.fabric import Fabric

NUM_STEPS_DEFAULT = 1000


def train_torch_ddp(
    rank,
    world_size,
    device=torch.device("cpu"),
    num_steps=NUM_STEPS_DEFAULT,
    batch_size=4,
):
    make_deterministic()

    os.environ["LOCAL_RANK"] = str(rank)
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    model = ConvNet().to(device)
    initial_state_dict = deepcopy(model.state_dict())

    ddp_model = DistributedDataParallel(model.to(device), device_ids=([rank] if device.type == "cuda" else None))

    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size * world_size), batch_size=batch_size)
    sampler = DistributedSampler(dataloader.dataset, rank=rank, num_replicas=world_size, drop_last=False, shuffle=False)
    dataloader = DataLoader(dataloader.dataset, sampler=sampler, batch_size=batch_size)
    optimizer = model.get_optimizer()
    loss_fn = model.get_loss_function()

    ddp_model.train()
    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(num_steps):
        t0 = time.perf_counter()

        inputs, labels = next(iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

    # check that the model has changed
    assert not is_state_dict_equal(initial_state_dict, ddp_model.module.state_dict())

    return ddp_model.module.state_dict(), torch.tensor(iteration_timings)


def train_fabric_ddp(fabric, num_steps=NUM_STEPS_DEFAULT, batch_size=4):
    make_deterministic()

    model = ConvNet()
    initial_state_dict = deepcopy(model.state_dict())

    optimizer = model.get_optimizer()
    model, optimizer = fabric.setup(model, optimizer)

    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size * fabric.world_size), batch_size=batch_size)
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


@RunIf(standalone=True)
# @pytest.mark.flaky(reruns=3)
@pytest.mark.usefixtures("reset_deterministic_algorithm", "reset_cudnn_benchmark")
@pytest.mark.parametrize(
    "accelerator, devices",
    [
        ("cpu", 2),
        pytest.param("gpu", 2, marks=RunIf(min_cuda_gpus=2)),
    ],
)
def test_parity_ddp(accelerator, devices):
    # Train with Fabric
    fabric = Fabric(accelerator=accelerator, strategy="ddp", devices=devices)
    fabric.launch()
    state_dict_fabric, timings_fabric = train_fabric_ddp(fabric)

    # Train with raw PyTorch
    state_dict_torch, timings_torch = train_torch_ddp(
        rank=fabric.global_rank,
        world_size=fabric.world_size,
        device=fabric.device,
    )

    # Compare the final weights
    assert is_state_dict_equal(state_dict_torch, state_dict_fabric)

    # Compare the time per iteration
    # Drop measurements of the first iterations, as they may be slower than others
    # The median is more robust to outliers than the mean
    # Given relative and absolute tolerances, we want to satisfy: |torch – fabric| < RTOL * |torch| + ATOL
    assert torch.isclose(torch.median(timings_torch[3:]), torch.median(timings_fabric[3:]), rtol=1e-3, atol=1e-3)
