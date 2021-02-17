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

Example script of running the experimental DDP Sequential Plugin.
This script splits a convolutional model onto multiple GPUs, whilst using the internal built in balancer
to balance across your GPUs.

To run:
python conv_model_sequential_example.py --accelerator ddp --gpus 4 --max_epochs 1  --batch_size 256 --use_rpc_sequential
"""
import math
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.plugins import RPCSequentialPlugin
from pytorch_lightning.utilities import _BOLTS_AVAILABLE, _FAIRSCALE_PIPE_AVAILABLE

if _BOLTS_AVAILABLE:
    import pl_bolts
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

#####################
#      Modules      #
#####################


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


###############################
#       LightningModule       #
###############################


class LitResnet(pl.LightningModule):
    """
    >>> LitResnet()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitResnet(
      (sequential_module): Sequential(...)
    )
    """

    def __init__(self, lr=0.05, batch_size=32, manual_optimization=False):
        super().__init__()

        self.save_hyperparameters()
        self.sequential_module = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
        self._example_input_array = torch.randn((1, 3, 32, 32))
        self._manual_optimization = manual_optimization
        if self._manual_optimization:
            self.training_step = self.training_step_manual

    def forward(self, x):
        out = self.sequential_module(x)
        return F.log_softmax(out, dim=-1)

    def training_step_manual(self, batch, batch_idx):
        opt = self.optimizers()

        def closure():
            x, y = batch
            logits = self.forward(x)
            loss = F.nll_loss(logits, y)
            self.manual_backward(loss, opt)
            self.log('train_loss', loss, prog_bar=True)

        opt.step(closure=closure)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log('Training Loss', loss)
        return loss

    def _evaluate(self, batch, batch_idx, stage=None):
        x, y = batch
        out = self.forward(x)
        logits = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

        return loss, acc

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, batch_idx, 'val')[0]

    def test_step(self, batch, batch_idx):
        loss, acc = self._evaluate(batch, batch_idx, 'test')
        self.log_dict({'test_loss': loss, 'test_acc': acc})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=math.ceil(45000 / self.hparams.batch_size)
                ),
                'interval': 'step',
            }
        }

    @property
    def automatic_optimization(self) -> bool:
        return not self._manual_optimization


#################################
#     Instantiate Data Module   #
#################################


def instantiate_datamodule(args):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ])

    cifar10_dm = pl_bolts.datamodules.CIFAR10DataModule(
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    return cifar10_dm


if __name__ == "__main__":
    cli_lightning_logo()

    assert _BOLTS_AVAILABLE, "Bolts is required for this example, install it via pip install pytorch-lightning-bolts"
    assert _FAIRSCALE_PIPE_AVAILABLE, "FairScale and PyTorch 1.6 is required for this example."

    parser = ArgumentParser(description="Pipe Example")
    parser.add_argument("--use_rpc_sequential", action="store_true")
    parser = Trainer.add_argparse_args(parser)
    parser = pl_bolts.datamodules.CIFAR10DataModule.add_argparse_args(parser)
    args = parser.parse_args()

    cifar10_dm = instantiate_datamodule(args)

    plugins = None
    if args.use_rpc_sequential:
        plugins = RPCSequentialPlugin()

    model = LitResnet(batch_size=args.batch_size, manual_optimization=not args.automatic_optimization)

    trainer = pl.Trainer.from_argparse_args(args, plugins=[plugins] if plugins else None)
    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    if trainer.accelerator.rpc_enabled:
        # Called at the end of trainer to ensure all processes are killed
        trainer.training_type_plugin.exit_rpc_process()
