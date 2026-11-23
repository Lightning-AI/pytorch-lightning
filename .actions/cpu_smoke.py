"""Exercise the installed standalone packages on CPU."""

import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def smoke_fabric():
    from lightning_fabric import Fabric

    fabric = Fabric(accelerator="cpu", devices=1)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)
    before = model.module.weight.detach().clone()
    fabric.backward(model(torch.ones(1, 1)).sum())
    optimizer.step()
    assert not torch.equal(before, model.module.weight)


def smoke_pytorch():
    from pytorch_lightning import LightningModule, Trainer

    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            (inputs,) = batch
            return self.layer(inputs).sum()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    model = Model()
    before = model.layer.weight.detach().clone()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, DataLoader(TensorDataset(torch.ones(2, 1))))
    assert trainer.global_step == 1
    assert not torch.equal(before, model.layer.weight)


if __name__ == "__main__":
    package, torch_version = sys.argv[1:]
    assert torch.__version__.split(".")[:2] == torch_version.split("."), torch.__version__
    {"fabric": smoke_fabric, "pytorch": smoke_pytorch}[package]()
