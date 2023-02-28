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
from lightning.fabric.utilities.cloud_io import _atomic_save

from tests_fabric.parity.utils import precision_context, is_state_dict_equal, make_deterministic
from tests_fabric.parity.models import ConvNet


def train_torch(
    move_to_device: Callable,
    precision_context,
    num_steps=1,
    batch_size=4,
    checkpoint_dir=".",
):
    make_deterministic()
    model = ConvNet()
    model = move_to_device(model)
    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
    optimizer = model.get_optimizer()
    loss_fn = model.get_loss_function()

    model.train()
    iterator = iter(dataloader)
    for _ in range(num_steps):
        inputs, labels = next(iterator)
        inputs, labels = move_to_device(inputs), move_to_device(labels)
        optimizer.zero_grad()
        with precision_context():
            outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    _atomic_save(model.state_dict(), os.path.join(checkpoint_dir, "torch_model.pt"))


def train_torch_ddp(
    rank,
    world_size,
    device=torch.device("cpu"),
    num_steps=1,
    batch_size=4,
    checkpoint_dir=".",
):
    make_deterministic()

    os.environ["LOCAL_RANK"] = str(rank)
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    model = ConvNet().to(device)
    initial_state_dict = deepcopy(model.state_dict())

    ddp_model = DistributedDataParallel(model.to(device), device_ids=([rank] if device.type == "cuda" else None))

    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
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
        inputs, labels = move_to_device(inputs), move_to_device(labels)
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # check that the model has changed
    assert not is_state_dict_equal(initial_state_dict, ddp_model.module.state_dict())

    if rank == 0:
        _atomic_save(ddp_model.module.state_dict(), os.path.join(checkpoint_dir, "torch_model.pt"))


class FabricRunner(Fabric):
    def run(self, num_steps=1, batch_size=4, checkpoint_dir="."):
        make_deterministic()

        model = ConvNet()
        initial_state_dict = deepcopy(model.state_dict())

        optimizer = model.get_optimizer()
        model, optimizer = self.setup(model, optimizer)

        dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
        dataloader = self.setup_dataloaders(dataloader)
        loss_fn = model.get_loss_function()

        model.train()
        iterator = iter(dataloader)
        for _ in range(num_steps):
            inputs, labels = next(iterator)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            self.backward(loss)
            optimizer.step()

        # check that the model has changed
        assert not is_state_dict_equal(initial_state_dict, model.state_dict())

        if self.global_rank == 0:
            _atomic_save(model.state_dict(), os.path.join(checkpoint_dir, "fabric_model.pt"))


@pytest.mark.parametrize(
    "precision, accelerator",
    [
        (32, "cpu"),
        pytest.param(32, "gpu", marks=RunIf(min_cuda_gpus=1)),
        # pytest.param(16, "gpu", marks=RunIf(min_cuda_gpus=1)),  # TODO: requires GradScaler
        pytest.param("bf16", "gpu", marks=RunIf(min_cuda_gpus=1, bf16_cuda=True)),
        pytest.param(32, "mps", marks=RunIf(mps=True)),
    ],
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_boring_fabric_model_single_device(precision, accelerator, tmpdir):
    fabric = FabricRunner(precision=precision, accelerator=accelerator, devices=1)
    fabric.run(checkpoint_dir=tmpdir)

    precision_ctx = partial(precision_context, precision=precision, accelerator=accelerator)
    train_torch(fabric.to_device, precision_context=fabric.autocast, checkpoint_dir=tmpdir)

    fabric_state_dict = torch.load(os.path.join(tmpdir, "fabric_model.pt"))
    torch_state_dict = torch.load(os.path.join(tmpdir, "torch_model.pt"))
    assert is_state_dict_equal(torch_state_dict, fabric_state_dict)


@RunIf(standalone=True)
@pytest.mark.parametrize(
    "precision, strategy, devices, accelerator",
    [
        (32, "ddp", 2, "cpu"),
        pytest.param(32, "ddp", 2, "gpu", marks=RunIf(min_cuda_gpus=2)),
    ],
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_boring_fabric_model_ddp(precision, strategy, devices, accelerator, tmpdir):
    fabric = FabricRunner(precision=precision, strategy=strategy, devices=devices, accelerator=accelerator)
    fabric.run(checkpoint_dir=tmpdir)

    with precision_context(precision, accelerator):
        train_torch_ddp(
            rank=fabric.global_rank, world_size=fabric.world_size, device=fabric.device, checkpoint_dir=tmpdir
        )

    tmpdir = fabric.broadcast(tmpdir)

    fabric_state_dict = torch.load(os.path.join(tmpdir, "fabric_model.pt"))
    torch_state_dict = torch.load(os.path.join(tmpdir, "torch_model.pt"))
    assert is_state_dict_equal(torch_state_dict, fabric_state_dict)
