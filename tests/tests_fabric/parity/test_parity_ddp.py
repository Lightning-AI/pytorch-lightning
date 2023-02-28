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
from copy import deepcopy
from functools import partial
from typing import Callable

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp
import torch.nn.functional
from lightning_utilities.core.apply_func import apply_to_collection
from tests_fabric.helpers.runif import RunIf
from torch import nn, Tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from unittest import mock

from lightning.fabric.fabric import Fabric
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.utilities.apply_func import move_data_to_device

from tests_fabric.parity.utils import precision_context, is_state_dict_equal, make_deterministic
from tests_fabric.parity.models import ConvNet

NUM_STEPS_DEFAULT = 100


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
    sampler = DistributedSampler(
        dataloader.dataset, rank=rank, num_replicas=world_size, drop_last=False, shuffle=False
    )
    dataloader = DataLoader(dataloader.dataset, sampler=sampler)
    optimizer = model.get_optimizer()
    loss_fn = model.get_loss_function()

    ddp_model.train()
    iterator = iter(dataloader)
    for _ in range(num_steps):
        inputs, labels = next(iterator)
        inputs, labels = inputs.to(device), inputs.to(labels)
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # check that the model has changed
    assert not is_state_dict_equal(initial_state_dict, ddp_model.module.state_dict())

    return ddp_model.module.state_dict()


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
    iterator = iter(dataloader)
    for _ in range(num_steps):
        inputs, labels = next(iterator)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

    # check that the model has changed
    assert not is_state_dict_equal(initial_state_dict, model.state_dict())

    return model.state_dict()


@RunIf(standalone=True)
@pytest.mark.parametrize(
    "precision, strategy, devices, accelerator",
    [
        (32, "ddp", 2, "cpu"),
        pytest.param(32, "ddp", 2, "gpu", marks=RunIf(min_cuda_gpus=2)),
    ],
)
def test_parity_ddp(precision, strategy, devices, accelerator, tmpdir):
    fabric = Fabric(precision=precision, strategy=strategy, devices=devices, accelerator=accelerator)
    fabric.launch()

    fabric_state_dict = train_fabric_ddp(fabric)
    torch_state_dict = train_torch_ddp(
        rank=fabric.global_rank, world_size=fabric.world_size, device=fabric.device
    )

    assert is_state_dict_equal(torch_state_dict, fabric_state_dict)
