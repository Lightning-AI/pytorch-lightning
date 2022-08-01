import os
import shutil
from time import sleep

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


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
        sleep(1)
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        print()
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 10), batch_size=2)

    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=1,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=ModelCheckpoint(monitor="train_loss", save_top_k=-1, every_n_train_steps=1)
    )
    trainer.fit(model, train_dataloaders=train_data)

    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=3,
        enable_model_summary=False,
        enable_progress_bar=True,
        callbacks=ModelCheckpoint(monitor="train_loss", save_top_k=-1, every_n_train_steps=1)
    )
    trainer.fit(model, train_dataloaders=train_data, ckpt_path="lightning_logs/version_0/checkpoints/epoch=0-step=3.ckpt")


if __name__ == "__main__":
    run()
