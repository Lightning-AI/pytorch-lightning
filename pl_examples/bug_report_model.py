import os
import random

import numpy
import torch
from torch.utils.data import Dataset

from pytorch_lightning import LightningModule, Trainer, seed_everything

import numpy as np
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __getitem__(self, index):
        return np.random.randint(0, 10, 3)

    def __len__(self):
        return 16


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self.layer(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        print(batch)
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def run():

    # fake data
    train_data = torch.utils.data.DataLoader(RandomDataset(), batch_size=2, num_workers=2)
    #
    # def worker_fn(worker_id):
    #     worker_seed = torch.initial_seed() % (2 ** 32)
    #     numpy.random.seed(worker_seed)
    #     random.seed(worker_seed)

    # train_data.worker_init_fn = worker_fn

    # model
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=4,
        max_epochs=2,
        weights_summary=None,
        # reload_dataloaders_every_epoch=True,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model, train_data)


if __name__ == '__main__':
    seed_everything(1)
    run()
