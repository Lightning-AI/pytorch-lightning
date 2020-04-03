import os
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from tests.base.datasets import TestingMNIST

try:
    from test_tube import HyperOptArgumentParser
except ImportError:
    # TODO: this should be discussed and moved out of this package
    raise ImportError('Missing test-tube package.')

from pytorch_lightning.core.lightning import LightningModule


class DictHparamsModel(LightningModule):

    def __init__(self, hparams: Dict):
        super().__init__()
        self.hparams = hparams
        self.l1 = torch.nn.Linear(hparams.get('in_features'), hparams['out_features'])

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(TestingMNIST(train=True, download=True), batch_size=16)


class TestModelBase(LightningModule):
    """Base LightningModule for testing. Implements only the required interface."""

    def __init__(self, hparams, force_remove_distributed_sampler: bool = False):
        """Pass in parsed HyperOptArgumentParser to the model."""
        # init superclass
        super().__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand(5, 28 * 28)

        # remove to test warning for dist sampler
        self.force_remove_distributed_sampler = force_remove_distributed_sampler

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """Layout model."""
        self.c_d1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.out_features)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """No special modification required for lightning, define as you normally would."""
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)

        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)

        return logits

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Lightning calls this inside the training loop"""
        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        # alternate possible outputs to test
        if self.trainer.batch_idx % 1 == 0:
            output = OrderedDict({
                'loss': loss_val,
                'progress_bar': {'some_val': loss_val * loss_val},
                'log': {'train_some_val': loss_val * loss_val},
            })

            return output
        if self.trainer.batch_idx % 2 == 0:
            return loss_val

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        # try no scheduler for this model (testing purposes)
        if self.hparams.optimizer_name == 'lbfgs':
            optimizer = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        _ = TestingMNIST(root=self.hparams.data_root, train=True, download=True)

    def _dataloader(self, train):
        # init data generators
        dataset = TestingMNIST(root=self.hparams.data_root, train=train, download=False)

        # when using multi-node we need to add the datasampler
        batch_size = self.hparams.batch_size

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )

        return loader
