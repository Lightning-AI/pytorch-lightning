"""
Example template for defining a system.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule


class LightningTemplateModel(LightningModule):
    """
    Sample model to show how to define a template.

    Example:

        >>> # define simple Net for MNIST dataset
        >>> params = dict(
        ...     drop_prob=0.2,
        ...     batch_size=2,
        ...     in_features=28 * 28,
        ...     learning_rate=0.001 * 8,
        ...     optimizer_name='adam',
        ...     data_root='./datasets',
        ...     out_features=10,
        ...     hidden_dim=1000,
        ... )
        >>> from argparse import Namespace
        >>> hparams = Namespace(**params)
        >>> model = LightningTemplateModel(hparams)
    """

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
        # init superclass
        super().__init__()
        self.hparams = hparams
        self.c_d1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.out_features)

    def forward(self, x):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        x = self.c_d1(x.view(x.size(0), -1))
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        self.mnist_train = MNIST(self.hparams.data_root, train=True, download=True, transform=transform)
        self.mnist_test = MNIST(self.hparams.data_root, train=False, download=True, transform=transform)

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=4)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=4)

    def test_dataloader(self):
        log.info('Test data loader called.')
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=4)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through `self.hparams`.
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=50000, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser
