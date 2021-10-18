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
from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Generator

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.imports import _TORCH_BFLOAT_AVAILABLE
from tests.helpers.boring_model import RandomDataset
from tests.helpers.runif import RunIf


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


def configure_optimizers(module: nn.Module):
    return torch.optim.SGD(module.parameters(), lr=0.001)


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


class LiteTrainer(LightningLite):
    def run(self, model: nn.Module, train_dataloader: DataLoader, num_epochs: int = 10):
        optimizer = configure_optimizers(model)
        model, optimizer = self.setup(model=model, optimizers=optimizer)
        train_dataloader = self.setup_dataloaders(train_dataloader)

        model.train()
        for _ in range(num_epochs):
            for batch in train_dataloader:
                batch = self.to_device(batch)
                optimizer.zero_grad()
                loss = model(batch)
                self.backward(loss)
                optimizer.step()


@contextmanager
def precision_context(precision, accelerator) -> Generator[None, None, None]:
    if precision == 32:
        yield
        return
    if precision == 16 and accelerator == "gpu":
        with torch.cuda.amp.autocast():
            yield
    elif accelerator == "cpu":
        with torch.cpu.amp.autocast(dtype=torch.float16 if precision == 16 else torch.bfloat16):
            yield
    else:
        with torch.cuda.amp.autocast():
            yield


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    "precision, strategy, devices, accelerator",
    [
        pytest.param(32, None, 1, "gpu"),
        pytest.param(16, None, 1, "gpu"),
        pytest.param(
            "bf16",
            None,
            1,
            "gpu",
            marks=pytest.mark.skipif(not _TORCH_BFLOAT_AVAILABLE, reason="bfloat16 isn't available."),
        ),
        pytest.param(32, None, 1, "cpu"),
    ],
)
def test_boring_lite_model_single_device(precision, strategy, devices, accelerator, tmpdir):
    seed_everything(42)
    train_dataloader = DataLoader(RandomDataset(32, 64))
    model = BoringModel()
    num_epochs = 1
    state_dict = deepcopy(model.state_dict())

    lite = LiteTrainer(precision=precision, strategy=strategy, devices=devices, accelerator=accelerator)
    lite.run(model, train_dataloader, num_epochs=num_epochs)
    lite_state_dict = model.state_dict()

    with precision_context(precision, accelerator):
        model.load_state_dict(state_dict)
        pure_state_dict = main(lite.to_device, model, train_dataloader, num_epochs=num_epochs)

    state_dict = apply_to_collection(state_dict, torch.Tensor, lite.to_device)
    for o_pure, w_lite in zip(state_dict.values(), lite_state_dict.values()):
        assert not torch.equal(o_pure, w_lite)

    for w_pure, w_lite in zip(pure_state_dict.values(), lite_state_dict.values()):
        assert torch.equal(w_pure, w_lite)
