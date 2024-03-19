# Copyright The Lightning AI team.
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
"""This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py.

Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.

Train on ImageNet with default parameters:

.. code-block: bash

    python imagenet.py fit --model.data_path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python imagenet.py --help
    python imagenet.py fit --help

"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.strategies import ParallelStrategy
from lightning.pytorch.utilities.model_helpers import get_torchvision_model
from torch.utils.data import Dataset
from torchmetrics import Accuracy


class ImageNetLightningModel(LightningModule):
    """
    >>> ImageNetLightningModel(data_path='missing')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ImageNetLightningModel(
      (model): ResNet(...)
    )
    """

    def __init__(
        self,
        data_path: str,
        arch: str = "resnet18",
        weights: Optional[str] = None,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        workers: int = 4,
    ):
        super().__init__()
        self.arch = arch
        self.weights = weights
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.model = get_torchvision_model(self.arch, weights=self.weights)
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        # ToDo: this number of classes hall be parsed when the dataset is loaded from folder
        self.train_acc1 = Accuracy(task="multiclass", num_classes=1000, top_k=1)
        self.train_acc5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.eval_acc1 = Accuracy(task="multiclass", num_classes=1000, top_k=1)
        self.eval_acc5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss_train = F.cross_entropy(output, target)
        self.log("train_loss", loss_train)
        # update metrics
        self.train_acc1(output, target)
        self.train_acc5(output, target)
        self.log("train_acc1", self.train_acc1, prog_bar=True)
        self.log("train_acc5", self.train_acc5, prog_bar=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)
        self.log(f"{prefix}_loss", loss_val)
        # update metrics
        self.eval_acc1(output, target)
        self.eval_acc5(output, target)
        self.log(f"{prefix}_acc1", self.eval_acc1, prog_bar=True)
        self.log(f"{prefix}_acc5", self.eval_acc5, prog_bar=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if isinstance(self.trainer.strategy, ParallelStrategy):
            # When using a single GPU per process and per `DistributedDataParallel`, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        if stage == "fit":
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_dir = os.path.join(self.data_path, "train")
            self.train_dataset = datasets.ImageFolder(
                train_dir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
            )
        # all stages will use the eval dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        val_dir = os.path.join(self.data_path, "val")
        self.eval_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    LightningCLI(
        ImageNetLightningModel,
        trainer_defaults={
            "max_epochs": 90,
            "accelerator": "auto",
            "devices": 1,
            "logger": False,
            "benchmark": True,
            "callbacks": [
                # the PyTorch example refreshes every 10 batches
                TQDMProgressBar(refresh_rate=10),
                # save when the validation top1 accuracy improves
                ModelCheckpoint(monitor="val_acc1", mode="max"),
            ],
        },
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )
