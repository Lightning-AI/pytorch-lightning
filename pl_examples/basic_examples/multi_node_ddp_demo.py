"""
Multi-node example (GPU)
"""
import os
from argparse import ArgumentParser

from pl_examples.models.lightning_template import LightningTemplateModel
from pytorch_lightning import Trainer, seed_everything

seed_everything(234)


def main(args):
    """Main training routine specific for this project."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(args)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp',
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == '__main__':
    run_cli()
