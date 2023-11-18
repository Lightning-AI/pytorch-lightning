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
"""This script will generate 2 traces: one for `training_step` and one for `validation_step`. The traces can be
visualized in 2 ways:

* With Chrome:
    1. Open Chrome and copy/paste this url: `chrome://tracing/`.
    2. Once tracing opens, click on `Load` at the top-right and load one of the generated traces.
* With PyTorch Tensorboard Profiler (Instructions are here: https://github.com/pytorch/kineto/tree/master/tb_plugin)
    1. pip install tensorboard torch-tb-profiler
    2. tensorboard --logdir={FOLDER}

"""

from os import path

import torch
import torchvision
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.profilers.pytorch import PyTorchProfiler
from lightning.pytorch.utilities.model_helpers import get_torchvision_model

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


class ModelToProfile(LightningModule):
    def __init__(self, name: str = "resnet18", automatic_optimization: bool = True):
        super().__init__()
        self.model = get_torchvision_model(name, weights="DEFAULT")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.automatic_optimization = automatic_optimization
        self.training_step = (
            self.automatic_optimization_training_step
            if automatic_optimization
            else self.manual_optimization_training_step
        )

    def automatic_optimization_training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def manual_optimization_training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        inputs = batch[0]
        return self.model(inputs)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


class CIFAR10DataModule(LightningDataModule):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def train_dataloader(self, *args, **kwargs):
        trainset = torchvision.datasets.CIFAR10(root=DATASETS_PATH, train=True, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)

    def val_dataloader(self, *args, **kwargs):
        valset = torchvision.datasets.CIFAR10(root=DATASETS_PATH, train=False, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(valset, batch_size=2, shuffle=True, num_workers=0)


def cli_main():
    cli = LightningCLI(
        ModelToProfile,
        CIFAR10DataModule,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "profiler": PyTorchProfiler(),
            "max_epochs": 1,
            "limit_train_batches": 15,
            "limit_val_batches": 15,
            "accelerator": "gpu",
        },
        run=False,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
