import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.examples.new_project_templates.lightning_module_template import LightningTemplateModel
from argparse import Namespace
from test_tube import Experiment
import numpy as np
import warnings
import torch
import os
import shutil

def get_model():
    # set up model with these hyperparams
    root_dir = os.path.dirname(os.path.realpath(__file__))
    hparams = Namespace(**{'drop_prob': 0.2,
                           'batch_size': 32,
                           'in_features': 28*28,
                           'learning_rate': 0.001*8,
                           'optimizer_name': 'adam',
                           'data_root': os.path.join(root_dir, 'mnist'),
                           'out_features': 10,
                           'hidden_dim': 1000})
    model = LightningTemplateModel(hparams)

    return model


def get_exp():
    # set up exp object without actually saving logs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    exp = Experiment(debug=True, save_dir=root_dir, name='tests_tt_dir')
    return exp


def clear_tt_dir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    tt_dir = os.path.join(root_dir, 'tests_tt_dir')
    if os.path.exists(tt_dir):
        shutil.rmtree(tt_dir)


def main():

    clear_tt_dir()
    model = get_model()

    trainer = Trainer(
        progress_bar=False,
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0, 1],
        distributed_backend='ddp',
        use_amp=True
    )

    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    clear_tt_dir()

if __name__ == '__main__':
    main()