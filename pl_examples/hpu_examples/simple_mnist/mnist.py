import os
import sys

import habana_frameworks.torch.core as htcore
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.callbacks import HPUStatsMonitor
from pytorch_lightning.plugins import HPUPrecisionPlugin
from pytorch_lightning.strategies.hpu import HPUStrategy
from pytorch_lightning.strategies.hpu_parallel import HPUParallelStrategy


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)

# TBD: import these keys from hmp
hmp_keys = ["level", "verbose", "bf16_ops", "fp32_ops"]
hmp_params = dict.fromkeys(hmp_keys)
hmp_params["level"] = "O1"
hmp_params["verbose"] = False
hmp_params["bf16_ops"] = "./pl_examples/hpu_examples/simple_mnist/ops_bf16_mnist.txt"
hmp_params["fp32_ops"] = "./pl_examples/hpu_examples/simple_mnist/ops_fp32_mnist.txt"

hpu_stats = HPUStatsMonitor(log_save_dir="habana_ptl_log", exp_name="mnist")

parallel_devices = 1
hpustrat_1 = HPUStrategy(
    device=torch.device("hpu"), precision_plugin=HPUPrecisionPlugin(precision=16, hmp_params=hmp_params)
)
hpustrat_8 = HPUParallelStrategy(
    parallel_devices=[torch.device("hpu")] * parallel_devices,
    precision_plugin=HPUPrecisionPlugin(precision=16, hmp_params=hmp_params),
)

# Initialize a trainer
trainer = pl.Trainer(
    strategy=hpustrat_8 if (parallel_devices == 8) else hpustrat_1,
    devices=parallel_devices,
    callbacks=[hpu_stats],
    max_epochs=1,
    default_root_dir=os.getcwd(),
    accelerator="hpu",
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)
