"""
Runs a model on the CPU on a single node.
"""
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_examples.models.lightning_template import LightningTemplateModel

pl.seed_everything(234)


def main(trainer_args, model_args):
    """ Main training routine specific for this project """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(**vars(model_args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer.from_argparse_args(trainer_args)

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
    trainer_parser = ArgumentParser(add_help=False)
    trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)
    trainer_args, model_args = trainer_parser.parse_known_args()

    # each LightningModule defines arguments relevant to it
    model_parser = ArgumentParser(add_help=False)
    parser = LightningTemplateModel.add_model_specific_args(model_parser, root_dir)
    model_args = parser.parse_args(model_args)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(trainer_args, model_args)
