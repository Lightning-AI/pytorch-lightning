import os
import unittest.mock

import torch
from torch.utils.data import DataLoader, Dataset

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
        print(batch.sum())
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def on_train_epoch_end(self):
        print("epoch ended")

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def on_load_checkpoint(self, checkpoint):
        pass


@unittest.mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=3,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model, train_dataloader=train_data)

    trainer.save_checkpoint("lightning_logs/auto.pt")

    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=3,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        max_epochs=3,
        weights_summary=None,
        resume_from_checkpoint="lightning_logs/auto.pt",
    )
    trainer.fit(model, train_dataloader=train_data)


if __name__ == "__main__":
    run()
