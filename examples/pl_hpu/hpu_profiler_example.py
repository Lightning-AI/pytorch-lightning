# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This script will generate 2 traces: one for `training_step` and one for `validation_step`. The traces can be
visualized in 2 ways:

* With Chrome:
    1. Open Chrome and copy/paste this url: `chrome://tracing/`.
    2. Once tracing opens, click on `Load` at the top-right and load one of the generated traces.
* With PyTorch Tensorboard Profiler (Instructions are here: https://github.com/pytorch/kineto/tree/master/tb_plugin)
    1. pip install tensorboard torch-tb-profiler
    2. tensorboard --logdir={FOLDER}
"""

import os
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.profilers.pytorch import PyTorchProfiler
from pytorch_lightning.utilities.imports import _KINETO_AVAILABLE

if _KINETO_AVAILABLE:
    from pytorch_lightning.profilers.hpu import HPUProfiler


class SimpleMNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.layer_1(x.view(x.size(0), -1)))

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SimpleMNISTDataModule(LightningDataModule):
    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=16, num_workers=1)


if __name__ == "__main__":
    data_module = SimpleMNISTDataModule()
    model = SimpleMNISTModel()

    if _KINETO_AVAILABLE:
        profiler = HPUProfiler(
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        accelerator = "hpu"
    else:
        profiler = PyTorchProfiler()
        accelerator = "cpu"
        warnings.warn(
            f"""_KINETO_AVAILABLE is {_KINETO_AVAILABLE}. Continuing with
                      profiler="PyTorchProfiler"
                      accelerator="{accelerator}" """
        )

    trainer = Trainer(
        profiler=profiler,
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        limit_train_batches=16,
        limit_val_batches=16,
    )

    trainer.fit(model, data_module)
