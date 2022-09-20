from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from lightning_lite import LightningLite


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int) -> None:
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class BoringLite(LightningLite):

    def get_model(self) -> Module:
        return nn.Linear(32, 2)

    def get_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def step(self, model: Module, batch: Any) -> Tensor:
        output = model(batch)
        loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
        return loss

    def after_backward(self, model: Module) -> None:
        pass

    def after_optimizer_step(self, model: Module, optimizer: Optimizer) -> None:
        pass

    def run(self) -> None:
        model = self.get_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dataloader = self.get_dataloader()

        model, optimizer = self.setup(model, optimizer)
        dataloader = self.setup_dataloaders(dataloader)

        data_iter = iter(dataloader)
        batch = next(data_iter)
        loss = self.step(model, batch)
        self.backward(loss)
        self.after_backward(model)
        optimizer.step()
        self.after_optimizer_step(model, optimizer)
        optimizer.zero_grad()
