import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Dict

try:
    from test_tube import HyperOptArgumentParser
except ImportError:
    # TODO: this should be discussed and moved out of this package
    raise ImportError('Missing test-tube package.')

from pytorch_lightning.core.lightning import LightningModule

# TODO: remove after getting own MNIST
# TEMPORAL FIX, https://github.com/pytorch/vision/issues/1938
import urllib.request
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


class TestingMNIST(MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, num_samples=8000):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        # take just a subset of MNIST dataset
        self.data = self.data[:num_samples]
        self.targets = self.targets[:num_samples]


class DictHparamsModel(LightningModule):

    def __init__(self, hparams: Dict):
        super(DictHparamsModel, self).__init__()
        self.hparams = hparams
        self.l1 = torch.nn.Linear(hparams.get('in_features'), hparams['out_features'])

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(TestingMNIST(os.getcwd(), train=True, download=True,
                                       transform=transforms.ToTensor()), batch_size=32)


class TestModelBase(LightningModule):
    """
    Base LightningModule for testing. Implements only the required
    interface
    """

    def __init__(self, hparams, force_remove_distributed_sampler=False):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
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
        """
        Layout model
        :return:
        """
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
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
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
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)

        y_hat = self.forward(x)

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
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        _ = TestingMNIST(root=self.hparams.data_root, train=True,
                         transform=transform, download=True, num_samples=2000)

    def _dataloader(self, train):
        # init data generators
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = TestingMNIST(root=self.hparams.data_root, train=train,
                               transform=transform, download=False, num_samples=2000)

        # when using multi-node we need to add the datasampler
        batch_size = self.hparams.batch_size

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )

        return loader
