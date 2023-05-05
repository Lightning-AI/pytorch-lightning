from typing import Any, Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset

from lightning.fabric import Fabric
from lightning.fabric.strategies.fsdp import FSDPStrategy


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int) -> None:
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int) -> None:
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(self.count):
            yield torch.randn(self.size)


class BoringFabric(Fabric):
    def get_model(self) -> Module:
        return nn.Linear(32, 2)

    def get_optimizer(self, module: Module) -> Optimizer:
        return torch.optim.SGD(module.parameters(), lr=0.1)

    def get_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(32, 64))

    def step(self, model: Module, batch: Any) -> Tensor:
        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))

    def after_backward(self, model: Module, optimizer: Optimizer) -> None:
        pass

    def after_optimizer_step(self, model: Module, optimizer: Optimizer) -> None:
        pass

    def run(self) -> None:
        model = self.get_model()
        if isinstance(self.strategy, FSDPStrategy):
            model = self.setup_module(model)
            optimizer = self.get_optimizer(model)
            optimizer = self.setup_optimizers(optimizer)
        else:
            optimizer = self.get_optimizer(model)
            model, optimizer = self.setup(model, optimizer)

        dataloader = self.get_dataloader()
        dataloader = self.setup_dataloaders(dataloader)

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader

        model.train()

        data_iter = iter(dataloader)
        batch = next(data_iter)
        loss = self.step(model, batch)
        self.backward(loss)
        self.after_backward(model, optimizer)
        optimizer.step()
        self.after_optimizer_step(model, optimizer)
        optimizer.zero_grad()
