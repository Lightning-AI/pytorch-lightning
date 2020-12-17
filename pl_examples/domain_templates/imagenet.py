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
This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.

Train on ImageNet with default parameters:

.. code-block: bash

    python imagenet.py --data-path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python imagenet.py --help

"""
import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning.core import LightningModule


class ImageNetLightningModel(LightningModule):
    """
    >>> ImageNetLightningModel(data_path='missing')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ImageNetLightningModel(
      (model): ResNet(...)
    )
    """
    # pull out resnet names from torchvision models
    MODEL_NAMES = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )

    def __init__(
            self,
            data_path: str,
            arch: str = 'resnet18',
            pretrained: bool = False,
            lr: float = 0.1,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            batch_size: int = 4,
            workers: int = 2,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.model = models.__dict__[self.arch](pretrained=self.pretrained)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log('train_acc5', acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('val_loss', loss_val, on_step=True, on_epoch=True)
        self.log('val_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log('val_acc5', acc5, on_step=True, on_epoch=True)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 0.1 ** (epoch // 30)
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        train_dir = os.path.join(self.data_path, 'train')
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )
        return train_loader

    def val_dataloader(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        val_dir = os.path.join(self.data_path, 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        outputs = self.validation_epoch_end(*args, **kwargs)

        def substitute_val_keys(out):
            return {k.replace('val', 'test'): v for k, v in out.items()}

        outputs = {
            'test_loss': outputs['val_loss'],
            'progress_bar': substitute_val_keys(outputs['progress_bar']),
            'log': substitute_val_keys(outputs['log']),
        }
        return outputs

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            choices=ImageNetLightningModel.MODEL_NAMES,
                            help=('model architecture: ' + ' | '.join(ImageNetLightningModel.MODEL_NAMES)
                                  + ' (default: resnet18)'))
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        return parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.accelerator == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = ImageNetLightningModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--data-path', metavar='DIR', type=str,
                               help='path to dataset')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('--seed', type=int, default=42,
                               help='seed for initializing training.')
    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler="simple",
        deterministic=True,
        max_epochs=90,
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_lightning_logo()
    run_cli()
