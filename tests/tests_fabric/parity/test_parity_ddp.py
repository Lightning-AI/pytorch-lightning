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
from functools import partial
from typing import Callable

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp
import torch.nn.functional
from tests_fabric.helpers.runif import RunIf
from torch import nn, Tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from unittest import mock

from lightning.fabric.fabric import Fabric
from lightning.fabric.utilities.cloud_io import _atomic_save

from tests_fabric.parity.utils import precision_context, is_state_dict_equal, make_deterministic
from tests_fabric.parity.models import ConvNet

NUM_STEPS_DEFAULT = 2000


def train_torch_ddp(
    rank,
    world_size,
    device=torch.device("cpu"),
    num_steps=NUM_STEPS_DEFAULT,
    batch_size=4,
    checkpoint_dir=".",
):
    make_deterministic()

    os.environ["LOCAL_RANK"] = str(rank)
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    model = ConvNet()
    initial_state_dict = deepcopy(model.state_dict())

    ddp_model = DistributedDataParallel(model.to(device), device_ids=([rank] if device.type == "cuda" else None))

    dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
    sampler = DistributedSampler(
        dataloader.dataset, rank=rank, num_replicas=world_size, seed=1, drop_last=False, shuffle=False
    )
    dataloader = DataLoader(dataloader.dataset, sampler=sampler)
    optimizer = model.get_optimizer()
    loss_fn = model.get_loss_function()

    iteration_timings = []

    ddp_model.train()
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

    if rank == 0:
        state = dict(state_dict=ddp_model.module.state_dict(), iteration_timings=torch.tensor(iteration_timings))
        _atomic_save(state, os.path.join(checkpoint_dir, "torch_model.pt"))


class FabricRunner(Fabric):
    def run(self, num_steps=NUM_STEPS_DEFAULT, batch_size=4, checkpoint_dir="."):
        make_deterministic()

        model = ConvNet()
        initial_state_dict = deepcopy(model.state_dict())

        optimizer = model.get_optimizer()
        model, optimizer = self.setup(model, optimizer)

        dataloader = model.get_dataloader(dataset_size=(num_steps * batch_size), batch_size=batch_size)
        dataloader = self.setup_dataloaders(dataloader)
        loss_fn = model.get_loss_function()

        iteration_timings = []

        model.train()
        iterator = iter(dataloader)
        for _ in range(num_steps):
            t0 = time.perf_counter()

            inputs, labels = next(iterator)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            self.backward(loss)
            optimizer.step()

        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)

        # check that the model has changed
        assert not is_state_dict_equal(initial_state_dict, model.state_dict())

        if self.global_rank == 0:
            state = dict(state_dict=model.state_dict(), iteration_timings=torch.tensor(iteration_timings))
            _atomic_save(state, os.path.join(checkpoint_dir, "fabric_model.pt"))


@RunIf(standalone=True)
@pytest.mark.parametrize(
    "precision, strategy, devices, accelerator",
    [
        (32, "ddp", 2, "cpu"),
        pytest.param(32, "ddp", 2, "gpu", marks=RunIf(min_cuda_gpus=2)),
    ],
)
def test_parity_ddp(precision, strategy, devices, accelerator, tmpdir):
    fabric = FabricRunner(precision=precision, strategy=strategy, devices=devices, accelerator=accelerator)
    fabric.run(checkpoint_dir=tmpdir)

    train_torch_ddp(
        rank=fabric.global_rank, world_size=fabric.world_size, device=fabric.device, checkpoint_dir=tmpdir
    )

    tmpdir = fabric.broadcast(tmpdir)

    fabric_results = torch.load(os.path.join(tmpdir, "fabric_model.pt"))
    torch_results = torch.load(os.path.join(tmpdir, "torch_model.pt"))
    assert is_state_dict_equal(fabric_results["state_dict"], torch_results["state_dict"])

    timings_fabric = fabric_results["iteration_timings"]
    timings_torch = torch_results["iteration_timings"]
    # The median is more robust to outliers than the mean
    assert torch.isclose(torch.median(timings_torch), torch.median(timings_fabric), rtol=1e-4, atol=1e-4)
