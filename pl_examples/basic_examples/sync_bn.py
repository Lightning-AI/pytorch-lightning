"""
Sync-bn with DDP (GPU)
"""
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from tests.base.datamodules import TrialMNISTDataModule


pl.seed_everything(234)

class SyncBNModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.BatchNorm1d(28 * 28)

    def forward(self, x):

        with torch.no_grad:
            return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self(x)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def main(args, datamodule):
    """Main training routine specific for this project."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SyncBNModule()

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp',
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, datamodule)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # define datamodule and dataloader
    dm = TrialMNISTDataModule()
    train_dataloader = dm.train_dataloader()

    for idx, batch in enumerate(train_dataloader):
        x, y = batch



    # each LightningModule defines arguments relevant to it
    parser = SyncBNModule.add_model_specific_args(parent_parser, root_dir)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args, dm)


if __name__ == '__main__':
    run_cli()
