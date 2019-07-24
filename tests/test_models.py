import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.examples.new_project_templates.lightning_module_template import LightningTemplateModel
from argparse import Namespace
from test_tube import Experiment
import numpy as np
import warnings
import torch
import os

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


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


def assert_ok_acc(trainer):
    # this model should get 0.80+ acc
    assert trainer.tng_tqdm_dic['val_acc'] > 0.80, "model failed to get expected 0.80 validation accuracy"


def test_cpu_model():
    """
    Make sure model trains on CPU
    :return:
    """
    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4
    )

    result = trainer.fit(model)
    assert result == 1, 'cpu model failed to complete'

    assert_ok_acc(trainer)


def test_single_gpu_model():
    """
    Make sure single GPU works (DP mode)
    :return:
    """
    if not torch.cuda.is_available():
        warnings.warn('test_single_gpu_model cannot run. Rerun on a GPU node to run this test')
        return

    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0]
    )

    result = trainer.fit(model)

    assert result == 1, 'single gpu model failed to complete'
    assert_ok_acc(trainer)


def test_multi_gpu_model_dp():
    """
    Make sure DP works
    :return:
    """
    if not torch.cuda.is_available():
        warnings.warn('test_multi_gpu_model_dp cannot run. Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_multi_gpu_model_dp cannot run. Rerun on a node with 2+ GPUs to run this test')
        return

    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0, 1]
    )

    result = trainer.fit(model)

    assert result == 1, 'multi-gpu dp model failed to complete'
    assert_ok_acc(trainer)


def test_multi_gpu_model_ddp():
    """
    Make sure DDP works
    :return:
    """
    if not torch.cuda.is_available():
        warnings.warn('test_multi_gpu_model_ddp cannot run. Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_multi_gpu_model_ddp cannot run. Rerun on a node with 2+ GPUs to run this test')
        return

    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0, 1],
        distributed_backend='ddp'
    )

    result = trainer.fit(model)

    assert result == 1, 'multi-gpu ddp model failed to complete'
    assert_ok_acc(trainer)


def test_amp_gpu_ddp():
    """
    Make sure DDP + AMP work
    :return:
    """
    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_ddp cannot run. Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_ddp cannot run. Rerun on a node with 2+ GPUs to run this test')
        return

    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0, 1],
        distributed_backend='ddp',
        use_amp=True
    )

    result = trainer.fit(model)

    assert result == 1, 'amp + ddp model failed to complete'
    assert_ok_acc(trainer)


def test_amp_gpu_dp():
    """
    Make sure DP + AMP work
    :return:
    """
    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_dp cannot run. Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_dp cannot run. Rerun on a node with 2+ GPUs to run this test')
        return

    model = get_model()

    trainer = Trainer(
        experiment=get_exp(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        gpus=[0, 1],
        distributed_backend='dp',
        use_amp=True
    )

    result = trainer.fit(model)

    assert result == 1, 'amp + gpu model failed to complete'
    assert_ok_acc(trainer)


if __name__ == '__main__':
    pytest.main([__file__])
