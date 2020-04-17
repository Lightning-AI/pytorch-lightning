"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from pl_examples.models.lightning_template import LightningTemplateModel
from pl_examples.utils.loaders import AsynchronousLoader

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


class AsyncModel(LightningTemplateModel):
    def __dataloader(self, train):
        # this is neede when you want some info about dataset before binding to trainer
        self.prepare_data()
        # init data generators
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(root=self.hparams.data_root, train=train,
                        transform=transform, download=False)

        # when using multi-node (ddp) we need to add the  datasampler
        batch_size = self.hparams.batch_size

        loader = AsynchronousLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=8
        )

        return loader


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
