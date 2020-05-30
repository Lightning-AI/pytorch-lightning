"""
Example template for defining a system.
"""
import os

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class SuperLitModel(pl.LightningModule):
    """
    Sample model to show how to define a template.

    Example:
    """

    def __init__(self,
                 drop_prob: float = 0.2,
                 batch_size: int = 2,
                 in_features: int = 28 * 28,
                 learning_rate: float = 0.001 * 8,
                 optimizer_name: str = 'adam',
                 out_features: int = 10,
                 hidden_dim: int = 1000,
                 **kwargs
                 ):
        # init superclass
        super().__init__()
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.in_features = in_features
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.c_d1 = nn.Linear(in_features=self.in_features,
                              out_features=self.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hidden_dim,
                              out_features=self.out_features)

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

    def training_step(self, batch: Tensor, batch_idx: int):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # structure the return from the training loop
        step_result = pl.TrainResult(
            minimize=loss,
            checkpoint_on=loss,
            early_stop_on=loss,
        )

        step_result.log('train_loss', loss)
        return step_result

    def validation_step(self, batch: Tensor, batch_idx: int):
        # forward pass
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)

        result = pl.EvalResult()
        result.log('val_loss', val_loss)
        result.to_pbar('pbar_loss', val_loss)

        return result

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        parser.add_argument('--hidden_dim', default=5000, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--data_dir', default='.', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser
    import pytorch_lightning as pl

    # add trainer args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # add model args
    parser = SuperLitModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # init data, model
    mnist_train = MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    mnist_train = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=0)
    model = SuperLitModel(**vars(args))

    # init trainer
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(
        model,
        train_dataloader=mnist_train,
        val_dataloaders=mnist_train
    )
