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
from typing import Generator

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite
from tests.helpers.boring_model import RandomDataset


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


def configure_optimizers(module: nn.Module):
    return torch.optim.SGD(module.parameters(), lr=0.001)


def main(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int = 10):
    optimizer = configure_optimizers(model)

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []

        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                val_losses.append(model(batch))

        train_epoch_loss = torch.stack(train_losses).mean()
        val_epoch_loss = torch.stack(val_losses).mean()

        print(f"{epoch}/{num_epochs}| Train Epoch Loss: {torch.mean(train_epoch_loss)}")
        print(f"{epoch}/{num_epochs}| Valid Epoch Loss: {torch.mean(val_epoch_loss)}")

    return model.state_dict()


class LiteTrainer(LightningLite):
    def run(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int = 10):
        seed_everything(42)
        optimizer = configure_optimizers(model)
        model, optimizer = self.setup(model=model, optimizers=optimizer)
        train_dataloader, val_dataloader = self.setup_dataloaders(train_dataloader, val_dataloader)

        for _ in range(num_epochs):
            train_losses = []
            val_losses = []

            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                loss = model(batch)
                train_losses.append(loss)
                self.backward(loss)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    val_losses.append(model(batch))


@contextmanager
def precision_context(precision, use_cpu, dype) -> Generator[None, None, None]:
    if precision == 32:
        yield
        return
    if dype == torch.bfloat16 and not use_cpu:
        with torch.cuda.amp.autocast(dtype=dype):
            yield
    elif use_cpu:
        with torch.cpu.amp.autocast(dtype=dype):
            yield
    else:
        with torch.cuda.amp.autocast():
            yield


@pytest.mark.parametrize(
    "dype,use_cpu,precision",
    [pytest.param(torch.bfloat16, True, "bf16"), pytest.param(torch.float32, True, 32)],
)
def test_boring_lite_model_cpu(dype, use_cpu, precision, tmpdir):

    seed_everything(42)
    train_dataloader = DataLoader(RandomDataset(32, 64))
    val_dataloader = DataLoader(RandomDataset(32, 64))
    model = BoringModel()
    num_epochs = 10

    state_dict = deepcopy(model.state_dict())

    with precision_context(precision, use_cpu, dype):
        pure_state_dict = main(model, train_dataloader, val_dataloader, num_epochs=num_epochs)

    model.load_state_dict(state_dict)
    lite = LiteTrainer(precision=precision)
    lite.run(model, train_dataloader, val_dataloader, num_epochs=num_epochs)

    for w_pure, w_lite in zip(pure_state_dict.values(), model.state_dict().values()):
        assert torch.equal(w_pure, w_lite)
