"""
Runs a model on a single node on a single GPU using asynchronous memory transfer
This example overlaps training and data transfer to reduce the time the GPU spends waiting for data
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pl_examples.models.lightning_template import LightningTemplateModel
from pl_examples.utils.loaders import AsynchronousLoader

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


class AsyncModel(LightningTemplateModel):
    def __dataloader(self, train):
        return AsynchronousLoader(dataloader=loader, device=torch.device('cuda', 0))


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = AsyncModel(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    hyperparams.distributed_backend = 'dp'
    hyperparams.gpus = 1

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
