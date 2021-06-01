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
"""
This script will generate 2 traces: one for `training_step` and one for `validation_step`.
The traces can be visualized in 2 ways:
* With Chrome:
    1. Open Chrome and copy/paste this url: `chrome://tracing/`.
    2. Once tracing opens, click on `Load` at the top-right and load one of the generated traces.
* With PyTorch Tensorboard Profiler (Instructions are here: https://github.com/pytorch/kineto/tree/master/tb_plugin)
    1. pip install tensorboard torch-tb-profiler
    2. tensorboard --logdir={FOLDER}
"""

import sys

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as T

from pl_examples import cli_lightning_logo
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import LightningCLI

DEFAULT_CMD_LINE = (
    "--trainer.max_epochs=1",
    "--trainer.limit_train_batches=15",
    "--trainer.limit_val_batches=15",
    "--trainer.profiler=pytorch",
    f"--trainer.gpus={int(torch.cuda.is_available())}",
)


class ModelToProfile(LightningModule):

    def __init__(self, name: str = "resnet50"):
        super().__init__()
        self.model = getattr(models, name)(pretrained=True)
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


class CIFAR10DataModule(LightningDataModule):

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def train_dataloader(self, *args, **kwargs):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    def val_dataloader(self, *args, **kwargs):
        valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=0)


def cli_main():
    if len(sys.argv) == 1:
        sys.argv += DEFAULT_CMD_LINE

    LightningCLI(ModelToProfile, CIFAR10DataModule)


if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()
