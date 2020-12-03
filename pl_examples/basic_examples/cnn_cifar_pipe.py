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
import math
from argparse import ArgumentParser

import torch
import torch.distributed as torch_distrib
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.plugins.pipe_rpc_plugin import FAIRSCALE_AVAILABLE, PipeRpcPlugin
from pytorch_lightning.utilities import BOLT_AVAILABLE

if BOLT_AVAILABLE:
    import pl_bolts
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


"""
To run with Pipe
python benchmarks/cnn_cifar_pipe.py --use_pipe 1

To run without Pipe
python benchmarks/cnn_cifar_pipe.py --use_pipe 0
"""


#####################
#      Module       #
#####################


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvNN(nn.Module):

    def __init__(self):
        super(ConvNN, self).__init__()

        self.layers = nn.Sequential(
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

    def forward(self, x):
        return self.layers(x)

###############################
#       LightningModule       #
###############################


class LitResnet(pl.LightningModule):
    def __init__(self, lr=0.05, batch_size=32, use_pipe=False):
        super().__init__()

        self.save_hyperparameters()
        model = ConvNN()
        self.layers = model.layers
        self._example_input_array = torch.randn((1, 3, 32, 32))
        if use_pipe:
            self.training_step = self.training_step_pipe

    def forward(self, x):
        out = self.layers(x)
        return F.log_softmax(out, dim=-1)

    def training_step_pipe(self, batch, batch_idx):
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

    def evaluate(self, batch, batch_idx, stage=None):
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
        return self.evaluate(batch, batch_idx, 'val')[0]

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, batch_idx, 'test')
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
                    steps_per_epoch=math.ceil(45000 / self.hparams.batch_size)),
                'interval': 'step',
            }
        }


#################################
#     Instantiate Functions     #
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


def run(args):

    cifar10_dm = instantiate_datamodule(args)

    plugins = None
    if args.use_pipe:
        plugins = [PipeRpcPlugin(balance=[17, 10])]
        gpus = 2
        accelerator = "ddp"
    else:
        gpus = 1
        accelerator = None

    model = LitResnet(batch_size=args.batch_size, use_pipe=args.use_pipe)

    trainer = pl.Trainer(
        progress_bar_refresh_rate=20,
        max_epochs=2,
        gpus=gpus,
        logger=pl.loggers.TensorBoardLogger('lightning_logs/', name='resnet'),
        callbacks=[LearningRateMonitor(logging_interval='step'),
                   ModelCheckpoint(filename='{epoch:03d}', save_last=True)],
        accelerator=accelerator,
        plugins=plugins,
        limit_train_batches=2,
        limit_val_batches=2,
        automatic_optimization=not args.use_pipe,
    )
    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    if args.use_pipe:
        if torch_distrib.get_rank() == 0:
            torch.distributed.rpc.shutdown()


if __name__ == "__main__":
    if FAIRSCALE_AVAILABLE and BOLT_AVAILABLE:
        parser = ArgumentParser(description="Pipe Example")
        parser.add_argument("--use_pipe", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=32)
        parser = Trainer.add_argparse_args(parser)
        run(parser.parse_args())
