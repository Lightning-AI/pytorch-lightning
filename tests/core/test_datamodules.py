import pickle
from argparse import ArgumentParser

import torch
import pytest

from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule
from tests.base.develop_utils import reset_seed


def test_base_datamodule(tmpdir):
    dm = TrialMNISTDataModule()
    dm.prepare_data()
    dm.setup()


def test_dm_has_been_called(tmpdir):
    dm = TrialMNISTDataModule()
    assert dm.prepare_data.has_been_called is False
    assert dm.setup.has_been_called is False

    dm.prepare_data()
    assert dm.prepare_data.has_been_called is True
    assert dm.setup.has_been_called is False

    dm.setup()
    assert dm.prepare_data.has_been_called is True
    assert dm.setup.has_been_called is True


def test_dm_add_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = TrialMNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', './my_data'])
    assert args.data_dir == './my_data'


def test_dm_init_from_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = TrialMNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', './my_data'])
    dm = TrialMNISTDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()


def test_dm_pickle_after_init(tmpdir):
    dm = TrialMNISTDataModule()
    pickle.dumps(dm)


def test_dm_pickle_after_setup(tmpdir):
    dm = TrialMNISTDataModule()
    dm.prepare_data()
    dm.setup()
    pickle.dumps(dm)


def test_train_loop_only(tmpdir):
    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()
    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)
    assert result == 1
    assert trainer.callback_metrics['loss'] < 0.6


def test_train_val_loop_only(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()
    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)
    assert result == 1
    assert trainer.callback_metrics['loss'] < 0.6


def test_full_loop(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="test requires multi-GPU machine")
def test_full_loop_single_gpu(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        gpus=1
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_full_loop_dp(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        distributed_backend='dp',
        gpus=2
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_full_loop_ddp_spawn(tmpdir):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        distributed_backend='ddp_spawn',
        gpus=[0, 1]
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8
