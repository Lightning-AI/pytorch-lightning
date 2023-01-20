import os

import torch
from torch.utils.data import DataLoader, Dataset

from lightning_fabric import Fabric
from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    fabric = Fabric(accelerator="cpu", strategy="ddp", devices=4)
    fabric.launch()

    fabric.barrier()

    rank = torch.tensor([fabric.global_rank, fabric.global_rank * 2])

    fabric.print(fabric.all_gather(dict(a=rank, b=(rank + 1), c="nothing")))
    fabric.print(fabric.all_reduce(dict(a=rank, b=(rank + 1), c="nothing"), reduce_op="sum"))


if __name__ == "__main__":
    run()
