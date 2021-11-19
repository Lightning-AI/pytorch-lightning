# Copyright The PyTorch Lightning team.
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
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Callable, Generator

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp
import torch.nn.functional
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.lite import LightningLite
from pytorch_lightning.plugins.environments.lightning_environment import find_free_network_port
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.cloud_io import atomic_save
from tests.helpers.boring_model import RandomDataset
from tests.helpers.runif import RunIf


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


def configure_optimizers(module: nn.Module):
    return torch.optim.SGD(module.parameters(), lr=0.0001)


def main(
    move_to_device: Callable,
    model: nn.Module,
    train_dataloader: DataLoader,
    num_epochs: int = 10,
):
    model = move_to_device(model)
    optimizer = configure_optimizers(model)

    for _ in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = move_to_device(batch)
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

    return model.state_dict()


class LiteRunner(LightningLite):
    def run(self, model: nn.Module, train_dataloader: DataLoader, num_epochs: int = 10, tmpdir: str = None):
        optimizer = configure_optimizers(model)
        model, optimizer = self.setup(model, optimizer)
        train_dataloader = self.setup_dataloaders(train_dataloader)

        model.train()
        for _ in range(num_epochs):
            for batch in train_dataloader:
                batch = self.to_device(batch)
                optimizer.zero_grad()
                loss = model(batch)
                self.backward(loss)
                optimizer.step()

        if isinstance(self._strategy, DDPSpawnPlugin) and tmpdir and self.global_rank == 0:
            checkpoint_path = os.path.join(tmpdir, "model.pt")
            atomic_save(model.state_dict(), checkpoint_path)
            return checkpoint_path


@contextmanager
def precision_context(precision, accelerator) -> Generator[None, None, None]:
    if precision == 32:
        yield
        return
    if accelerator == "gpu":
        with torch.cuda.amp.autocast():
            yield
    elif accelerator == "cpu":
        with torch.cpu.amp.autocast():
            yield


@pytest.mark.parametrize(
    "precision, strategy, devices, accelerator",
    [
        pytest.param(32, None, 1, "cpu"),
        pytest.param(32, None, 1, "gpu", marks=RunIf(min_gpus=1)),
        pytest.param(16, None, 1, "gpu", marks=RunIf(min_gpus=1)),
        pytest.param("bf16", None, 1, "gpu", marks=RunIf(min_torch="1.10", min_gpus=1)),
    ],
)
def test_boring_lite_model_single_device(precision, strategy, devices, accelerator, tmpdir):
    LightningLite.seed_everything(42)
    train_dataloader = DataLoader(RandomDataset(32, 8))
    model = BoringModel()
    num_epochs = 1
    state_dict = deepcopy(model.state_dict())

    lite = LiteRunner(precision=precision, strategy=strategy, devices=devices, accelerator=accelerator)
    lite.run(model, train_dataloader, num_epochs=num_epochs)
    lite_state_dict = model.state_dict()

    with precision_context(precision, accelerator):
        model.load_state_dict(state_dict)
        pure_state_dict = main(lite.to_device, model, train_dataloader, num_epochs=num_epochs)

    state_dict = apply_to_collection(state_dict, torch.Tensor, lite.to_device)
    for w_pure, w_lite in zip(state_dict.values(), lite_state_dict.values()):
        assert not torch.equal(w_pure, w_lite)

    for w_pure, w_lite in zip(pure_state_dict.values(), lite_state_dict.values()):
        assert torch.equal(w_pure, w_lite)


def run(rank, model, train_dataloader, num_epochs, precision, accelerator, tmpdir):
    os.environ["LOCAL_RANK"] = str(rank)
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", rank=rank, world_size=2)

    to_device = partial(move_data_to_device, device=torch.device("cuda", rank))
    model = DistributedDataParallel(
        to_device(model),
        device_ids=[rank],
    )
    train_dataloader = DataLoader(
        train_dataloader.dataset,
        sampler=DistributedSampler(train_dataloader.dataset, rank=rank, num_replicas=2, seed=42, drop_last=False),
    )
    with precision_context(precision, accelerator):
        main(to_device, model, train_dataloader, num_epochs=num_epochs)

    if rank == 0:
        atomic_save(model.state_dict(), os.path.join(tmpdir, "model_spawn.pt"))


@pytest.mark.skipif(True, reason="Skipping as it takes 80 seconds.")
@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    "precision, strategy, devices, accelerator",
    [
        (32, "ddp_spawn", 2, "gpu"),
    ],
)
def test_boring_lite_model_ddp_spawn(precision, strategy, devices, accelerator, tmpdir):
    LightningLite.seed_everything(42)
    train_dataloader = DataLoader(RandomDataset(32, 8))
    model = BoringModel()
    num_epochs = 1
    state_dict = deepcopy(model.state_dict())

    lite = LiteRunner(precision=precision, strategy=strategy, devices=devices, accelerator=accelerator)
    checkpoint_path = lite.run(model, train_dataloader, num_epochs=num_epochs, tmpdir=tmpdir)
    spawn_model_state_dict = torch.load(checkpoint_path)

    for w_pure, w_lite in zip(state_dict.values(), spawn_model_state_dict.values()):
        assert not torch.equal(w_pure.cpu(), w_lite.cpu())

    model.load_state_dict(state_dict)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_network_port())
    mp.spawn(run, args=(model, train_dataloader, num_epochs, precision, accelerator, tmpdir), nprocs=2)
    spawn_pure_model_state_dict = torch.load(os.path.join(tmpdir, "model_spawn.pt"))

    for w_pure, w_lite in zip(spawn_pure_model_state_dict.values(), spawn_model_state_dict.values()):
        assert torch.equal(w_pure.cpu(), w_lite.cpu())


@RunIf(min_gpus=2, special=True)
@pytest.mark.parametrize(
    "precision, strategy, devices, accelerator",
    [
        (32, "ddp", 2, "gpu"),
    ],
)
def test_boring_lite_model_ddp(precision, strategy, devices, accelerator, tmpdir):
    LightningLite.seed_everything(42)
    train_dataloader = DataLoader(RandomDataset(32, 4))
    model = BoringModel()
    num_epochs = 1
    state_dict = deepcopy(model.state_dict())

    lite = LiteRunner(precision=precision, strategy=strategy, devices=devices, accelerator=accelerator)
    lite.run(model, train_dataloader, num_epochs=num_epochs, tmpdir=tmpdir)

    lite_model_state_dict = model.state_dict()

    for w_pure, w_lite in zip(state_dict.values(), lite_model_state_dict.values()):
        assert not torch.equal(w_pure.cpu(), w_lite.cpu())

    LightningLite.seed_everything(42)
    train_dataloader = DataLoader(RandomDataset(32, 4))
    model = BoringModel()
    run(lite.global_rank, model, train_dataloader, num_epochs, precision, accelerator, tmpdir)
    pure_model_state_dict = model.state_dict()

    for w_pure, w_lite in zip(pure_model_state_dict.values(), lite_model_state_dict.values()):
        assert torch.equal(w_pure.cpu(), w_lite.cpu())
