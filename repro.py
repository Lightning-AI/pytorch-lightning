import os

import lightning.pytorch as pl
import timm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torchvision.datasets import CIFAR10
import time

epochs = 5
epoch_size = 100


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("rexnet_150", pretrained=True, num_classes=10)
        self.t0 = 0
        self.epoch_time = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def on_train_start(self):
        # torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)
        print(os.environ.get("OMP_NUM_THREADS"))
        print(torch.get_num_threads(), torch.get_num_interop_threads())

    def on_train_epoch_start(self):
        self.t0 = time.perf_counter()

    def on_train_epoch_end(self):
        self.epoch_time.append(time.perf_counter() - self.t0)

    def on_train_end(self):
        avg_epoch_time = sum(self.epoch_time) / len(self.epoch_time)
        it_per_sec = epochs * epoch_size / sum(self.epoch_time)
        self.trainer.print(f"avg time per epoch: {avg_epoch_time:.3f}")
        self.trainer.print(f"it/s: {it_per_sec:.3f}")

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        return F.cross_entropy(logits, y)


def run():
    transform = tfs.Compose([tfs.Resize((224, 224)), tfs.ToTensor()])
    dataset = CIFAR10(".", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64)
    model = LitModel()
    trainer = pl.Trainer(max_steps=5, max_epochs=epochs, limit_train_batches=epoch_size, accelerator="cuda", devices=2, strategy="ddp")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    run()
