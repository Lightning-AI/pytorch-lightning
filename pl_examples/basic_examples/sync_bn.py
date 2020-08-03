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
EPSILON = 1e-12


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
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp',
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        sync_bn_backend=args.sync_bn_backend,
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
    dm = TrialMNISTDataModule()
    dm.prepare_data()
    dm.setup()

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
    dm.setup()

    # each LightningModule defines arguments relevant to it
    parser = pl.Trainer.add_argparse_args(parent_parser)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args, dm, outputs)


if __name__ == '__main__':
    run_cli()
