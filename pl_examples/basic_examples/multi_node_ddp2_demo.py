"""
Multi-node example (GPU)
"""
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_examples.models.lightning_template import LightningTemplateModel

pl.seed_everything(234)


def main(model_args):
    """ Main training routine specific for this project """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(model_args)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=2,
        num_nodes=2,
        distributed_backend='ddp2'
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    model_args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(model_args)
