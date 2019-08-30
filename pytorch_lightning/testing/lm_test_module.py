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


class LightningTestModel(LightningTestModelBase):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, force_remove_distributed_sampler=False, use_two_test_sets=False):
        super(LightningTestModel, self).__init__(hparams, force_remove_distributed_sampler)
        self.use_two_test_sets = use_two_test_sets

    def validation_step(self, data_batch, batch_i, dataloader_i):
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
        if batch_i % 1 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
            })
            return output
        if batch_i % 2 == 0:
            return val_acc

        if batch_i % 3 == 0:
            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc': val_acc,
                'test_dic': {'val_loss_a': loss_val}
            })
            return output
        if batch_i % 5 == 0:
            output = OrderedDict({
                f'val_loss_{dataloader_i}': loss_val,
                f'val_acc_{dataloader_i}': val_acc,
            })
            return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        tqdm_dic = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
        return tqdm_dic

    def test_step(self, data_batch, batch_i, dataloader_i):
        """
        Lightning calls this inside the validation loop
        :param data_batch:
        :return:
        """
        x, y = data_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_test = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)

        if self.on_gpu:
            test_acc = test_acc.cuda(loss_test.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_test = loss_test.unsqueeze(0)
            test_acc = test_acc.unsqueeze(0)

        # alternate possible outputs to test
        if batch_i % 1 == 0:
            output = OrderedDict({
                'test_loss': loss_test,
                'test_acc': test_acc,
            })
            return output
        if batch_i % 2 == 0:
            return test_acc

        if batch_i % 3 == 0:
            output = OrderedDict({
                'test_loss': loss_test,
                'test_acc': test_acc,
                'test_dic': {'test_loss_a': loss_test}
            })
            return output
        if batch_i % 5 == 0:
            output = OrderedDict({
                f'test_loss_{dataloader_i}': loss_test,
                f'test_acc_{dataloader_i}': test_acc,
            })
            return output

    def test_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from test_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()
        test_loss_mean = 0
        test_acc_mean = 0
        for output in outputs:
            test_loss = output['test_loss']

            # reduce manually when using dp
            if self.trainer.use_dp:
                test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss

            # reduce manually when using dp
            test_acc = output['test_acc']
            if self.trainer.use_dp:
                test_acc = torch.mean(test_acc)

            test_acc_mean += test_acc

        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)

        tqdm_dic = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
        return tqdm_dic

    def on_tng_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)

    @data_loader
    def val_dataloader(self):
        return [self._dataloader(train=False), self._dataloader(train=False)]

    @data_loader
    def test_dataloader(self):
        if self.use_two_test_sets:
            return [self._dataloader(train=False), self._dataloader(train=False)]
        return self._dataloader(train=False)
