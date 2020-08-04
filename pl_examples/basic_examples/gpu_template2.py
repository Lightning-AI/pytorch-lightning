"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything, Callback
from pl_examples.models.lightning_template import LightningTemplateModel

seed_everything(234)


class DebugCallback(Callback):

    def on_test_batch_end(self, trainer, pl_module):
        print('test_batch', trainer.global_rank)


def main(args):
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(
        args,
        distributed_backend='ddp',
        limit_train_batches=10,
        limit_val_batches=10,
        max_epochs=1,
        callbacks=[DebugCallback()],
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)
    trainer.test(model)


def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=2)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == '__main__':
    run_cli()
