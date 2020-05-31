"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from pl_examples.models.lightning_template import LightningTemplateModel

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(trainer_args, hparams):
    """ Main training routine specific for this project """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(**vars(hparams))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=trainer_args.epochs,
        gpus=trainer_args.gpus,
        distributed_backend=trainer_args.distributed_backend,
        precision=16 if trainer_args.use_16bit else 32,
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
    # parent_parser = ArgumentParser()
    parent_parser = ArgumentParser(parents=[parent_parser])

    # gpu args
    trainer_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many gpus'
    )
    trainer_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='ddp',
        help='supports three options dp, ddp, ddp2'
    )
    trainer_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )
    trainer_parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='number of training epochs'
    )

    # each LightningModule defines arguments relevant to it
    trainer_args = trainer_parser.parse_args()

    hparams_parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = hparams_parser.parse_args()
    print(hyperparams)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(trainer_args, hyperparams)
