import inspect
import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loops.batch.yield_loop import Yield, YieldLoop
from pytorch_lightning.plugins import DDPPlugin


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(Yield, LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(32, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.head = torch.nn.Linear(32, 2)

    # potential future directions
    # 1) yield loss + optimizer
    # 2) last statement must be a return
    # 3) yield loss + extras for step_end and epoch_end
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss0 = self.layer1(batch).sum()
        yield loss0

        print("yield 0")

        loss1 = self.layer2(batch).sum()

        print("yield 1")

        yield loss1

    def configure_optimizers(self):
        # scheduler dict?
        opt1 = torch.optim.SGD(self.layer1.parameters(), lr=0.1)
        opt2 = torch.optim.SGD(self.layer2.parameters(), lr=0.1)
        return opt1, opt2


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
        # gpus=1,
        # accelerator="ddp",
        # accelerator="ddp_cpu",
        # plugins=DDPPlugin(),
        # num_processes=1,
    )

    yield_batch_loop = YieldLoop()
    trainer.fit_loop.epoch_loop.connect(batch_loop=yield_batch_loop)

    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    run()
