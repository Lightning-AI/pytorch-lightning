"""
Example template for defining a system
"""
import os
from collections import OrderedDict
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule


class LightningTemplateModel(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(LightningTemplateModel, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand(5, 28 * 28)

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

    def training_step(self, batch, batch_idx):
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
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch
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
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
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
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'logs': tqdm_dict}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        # init data generators
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(root=self.hparams.data_root, train=train,
                        transform=transform, download=True)

        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        batch_size = self.hparams.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        print('training data loader called')
        return self.__dataloader(train=True)

    @pl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        return self.__dataloader(train=False)

    @pl.data_loader
    def test_dataloader(self):
        print('test data loader called')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
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
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser
