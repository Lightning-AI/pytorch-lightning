import os

import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from pytorch.cli import LightningCLI


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
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    import os
    os.environ["PL_TRAINER_ACCELERATOR"] = "cpu"
    os.environ["PL_TRAINER_DEVICES"] = "2"

    cli = LightningCLI(BoringModel, run=False, trainer_defaults={"max_epochs": 1})
    cli.trainer.fit(cli.model, train_data, val_data)


if __name__ == "__main__":
    run()
