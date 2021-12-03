import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, seed_everything, Trainer


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
        print(self.global_rank, self.global_step)
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
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    seed_everything(42)

    model = BoringModel()
    model_copy = deepcopy(model)
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 8
    trainer = Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        accelerator="cpu",
        gpus=2,
        strategy="ddp_spawn",
    )

    trainer.fit(model, train_data)

    for param, param_copy in zip(model.parameters(), model_copy.parameters()):
        assert not torch.equal(param.cpu().data, param_copy.data)


if __name__ == "__main__":
    run()
