"""
Sync-bn with DDP (GPU)
"""
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader


pl.seed_everything(234)
EPSILON = 1e-12


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './'):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup() 
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download only
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)


class SyncBNModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.outputs = None
        if 'outputs' in kwargs:
            self.outputs = kwargs['outputs']

        self.layer = nn.BatchNorm1d(28 * 28)

    def forward(self, x, batch_idx):

        with torch.no_grad():
            x = self.layer(x.view(x.size(0), -1))
            
            """
            print('######')
            print(self.trainer.local_rank)
            print('######')
            
            assert 1 == 0
            # onle for rank 0 process, check half outputs
            if self.outputs:
                assert abs(torch.sum)
            """

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self(x, batch_idx)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @staticmethod
    def add_model_specific_argument(parent_parser, root_dir):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])
        
        parser.add_argument('--nodes', default=1, type=int)
        parser.add_argument('--gpu', default=2, type=int)
        
        parser.add_argument('--epochs', default=1, type=int)
        parser.add_argument('--steps', default=3, type=int)
        
        parser.add_argument('--sync_bn', default='torch', type=str)
        
        return parser


def main(args, datamodule, outputs):
    """Main training routine specific for this project."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SyncBNModule(outputs=outputs)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=args.gpu,
        num_nodes=args.nodes,
        distributed_backend='ddp',
        max_epochs=args.epochs,
        max_steps=args.steps,
        sync_bn_backend=args.sync_bn,
        num_sanity_val_steps=0,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, datamodule)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # define datamodule and dataloader
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup(stage=None)

    train_dataloader = dm.train_dataloader()
    model = SyncBNModule()

    outputs = []
    for idx, batch in enumerate(train_dataloader):
        x, y = batch

        outputs.append(model.forward(x, idx))

        # get 3 steps
        if idx == 2:
            break

    outputs = [x.cuda() for x in outputs]

    # reset datamodule
    dm.setup(stage=None)

    # each LightningModule defines arguments relevant to it
    parser = SyncBNModule.add_model_specific_argument(parent_parser, root_dir=root_dir)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args, dm, outputs)


if __name__ == '__main__':
    run_cli()
