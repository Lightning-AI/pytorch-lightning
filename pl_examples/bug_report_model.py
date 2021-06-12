import logging
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class ToyTask(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def setup(self, stage: str):
        if stage == "test":
            return
        self.setup_model_and_optimizer()
        print("setup called")

    def setup_model_and_optimizer(self):
        self.model = ToyModel()
        self.optimizer = AdamW(
            self.model.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1.0e-08, weight_decay=0, amsgrad=False
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        targets = self.forward(batch["model_input"])
        loss = self.loss_fn(targets, batch["label"])

        # Log loss results per train step and per epoch
        self.log("loss", loss)

        # Tell Lightning to minimize loss
        return loss

    def configure_optimizers(self):
        return self.optimizer

    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    # self.setup_model_and_optimizer()


if __name__ == "__main__":
    task = ToyTask()

    dataset = [{"model_input": torch.randn(20, 10), "label": torch.randn(20, 5)} for _ in range(10)]

    train_dataloader = DataLoader(dataset, batch_size=None)
    val_dataloader = DataLoader(dataset, batch_size=None)

    model_checkpoint = ModelCheckpoint(
        save_last=True,
        every_n_val_epochs=1,
    )

    trainer = pl.Trainer(
        gpus=2,
        precision=16,
        max_epochs=3,
        progress_bar_refresh_rate=100,
        log_gpu_memory=None,
        reload_dataloaders_every_epoch=True,
        limit_train_batches=10,
        limit_val_batches=10,
        limit_test_batches=10,
        callbacks=[model_checkpoint],
    )

    results = trainer.fit(task, train_dataloader)

    print(model_checkpoint.last_model_path)

    trainer = pl.Trainer(
        gpus=2,
        precision=16,
        max_epochs=4,
        reload_dataloaders_every_epoch=True,
        limit_train_batches=10,
        limit_val_batches=10,
        limit_test_batches=10,
        callbacks=[model_checkpoint],
        resume_from_checkpoint=model_checkpoint.last_model_path,
    )
    trainer.fit(task, train_dataloader)
