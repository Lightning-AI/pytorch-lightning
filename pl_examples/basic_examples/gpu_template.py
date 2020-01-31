"""
Runs a model on a single node across N-gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from pl_examples.basic_examples.lightning_module_template import LightningTemplateModel

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    if hparams.evaluate_val or hparams.evaluate_test:
        assert hparams.checkpoint != '', 'Please specify checkpoint for evaluation'
        model.load_from_checkpoint(hparams.checkpoint)

        if hparams.evaluate_val:
            trainer.validate(model)

        if hparams.evaluate_test:
            trainer.test(model)
    else:
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
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )
    parent_parser.add_argument(
        '--evaluate_val',
        dest='evaluate_val',
        action='store_true',
        help='evaluate on validation set'
    )
    parent_parser.add_argument(
        '--evaluate_test',
        dest='evaluate_test',
        action='store_true',
        help='evaluate on test set'
    )
    parent_parser.add_argument(
        '--checkpoint',
        default='',
        type=str,
        help='checkpoint to be evaluated'
    )

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
