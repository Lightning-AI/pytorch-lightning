import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision import transforms
from test_tube import HyperOptArgumentParser

from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_lightning import data_loader

from .lm_test_module_base import LightningTestModelBase


class NoValEndTestModel(LightningTestModelBase):
    """
    Sample model to show how to define a template
    """

    @data_loader
    def val_dataloader(self):
        return self._dataloader(train=False)

    def validation_step(self, data_batch, batch_nb):
        """
        Lightning calls this inside the validation loop
        :param data_batch:
        :return:
        """
        x, y = data_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        # alternate possible outputs to test
        if batch_nb % 1 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
            })
            return output
        if batch_nb % 2 == 0:
            return val_acc

        if batch_nb % 3 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
                'test_dic': {'val_loss_a': loss_val}
            })
            return output
