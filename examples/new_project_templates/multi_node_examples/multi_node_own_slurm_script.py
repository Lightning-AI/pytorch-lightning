"""
Multi-node example (GPU)
"""
import os
import numpy as np
import torch

from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning import Trainer
from examples.new_project_templates.lightning_module_template import LightningTemplateModel

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(hparams)

    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------
    # init experiment
    exp = Experiment(
        name='test_exp',
        save_dir=hyperparams.log_dir,
        autosave=False,
        description='test demo'
    )

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        experiment=exp,
        gpus=8,
        nb_gpu_nodes=2
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # use current dir for logging
    root_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(root_dir, 'pt_lightning_demo_logs')

    parent_parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
    parent_parser.add_argument('--log_dir', type=str, default=log_dir,
                               help='where to save logs')

    # allow model to overwrite or extend args
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
