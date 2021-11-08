import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import sys

import habana_frameworks.torch.core as htcore

class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
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
hmp_params["bf16_ops"] = "./pytorch-lightning-fork/pl_examples/hpu_examples/simple_mnist/ops_bf16_mnist.txt"
hmp_params["fp32_ops"] = "./pytorch-lightning-fork/pl_examples/hpu_examples/simple_mnist/ops_fp32_mnist.txt"

# Initialize a trainer
trainer = pl.Trainer(hpus=1, max_epochs=1, precision=16, hmp_params=hmp_params)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)
