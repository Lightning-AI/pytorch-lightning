import pickle
from argparse import ArgumentParser
from unittest.mock import MagicMock

import pytest
import torch

from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule
from tests.base.develop_utils import reset_seed


def test_can_prepare_data(tmpdir):

    dm = TrialMNISTDataModule()
    trainer = Trainer()
    trainer.datamodule = dm

    # 1 no DM
    # prepare_data_per_node = True
    # local rank = 0   (True)
    trainer.prepare_data_per_node = True
    trainer.local_rank = 0
    assert trainer.can_prepare_data()

    # local rank = 1   (False)
    trainer.local_rank = 1
    assert not trainer.can_prepare_data()

    # prepare_data_per_node = False (prepare across all nodes)
    # global rank = 0   (True)
    trainer.prepare_data_per_node = False
    trainer.node_rank = 0
    trainer.local_rank = 0
    assert trainer.can_prepare_data()

    # global rank = 1   (False)
    trainer.node_rank = 1
    trainer.local_rank = 0
    assert not trainer.can_prepare_data()
    trainer.node_rank = 0
    trainer.local_rank = 1
    assert not trainer.can_prepare_data()

    # 2 dm
    # prepar per node = True
    # local rank = 0 (True)
    trainer.prepare_data_per_node = True
    trainer.local_rank = 0

    # is_overridden prepare data = True
    # has been called
    # False
    dm._has_prepared_data = True
    assert not trainer.can_prepare_data()

    # has not been called
    # True
    dm._has_prepared_data = False
    assert trainer.can_prepare_data()

    # is_overridden prepare data = False
    # True
    dm.prepare_data = None
    assert trainer.can_prepare_data()


def test_base_datamodule(tmpdir):
    dm = TrialMNISTDataModule()
    dm.prepare_data()
    dm.setup()


def test_base_datamodule_with_verbose_setup(tmpdir):
    dm = TrialMNISTDataModule()
    dm.prepare_data()
    dm.setup('fit')
    dm.setup('test')


def test_data_hooks_called(tmpdir):
    dm = TrialMNISTDataModule()
    assert dm.has_prepared_data is False
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.prepare_data()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.setup()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


def test_data_hooks_called_verbose(tmpdir):
    dm = TrialMNISTDataModule()
    assert dm.has_prepared_data is False
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.prepare_data()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.setup('fit')
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is False

    dm.setup('test')
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


def test_data_hooks_called_with_stage_kwarg(tmpdir):
    dm = TrialMNISTDataModule()
    dm.prepare_data()
    assert dm.has_prepared_data is True

    dm.setup(stage='fit')
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is False

    dm.setup(stage='test')
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


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

    # fit model
    result = trainer.fit(model, dm)
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

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1
    assert trainer.callback_metrics['loss'] < 0.6


def test_test_loop_only(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )
    trainer.test(model, datamodule=dm)


def test_full_loop(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
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
        gpus=1,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
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
        gpus=2,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_full_loop_ddp_spawn(tmpdir):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    seed_everything(1234)

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        weights_summary=None,
        distributed_backend='ddp_spawn',
        gpus=[0, 1],
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="test requires multi-GPU machine")
def test_dm_transfer_batch_to_device(tmpdir):
    class CustomBatch:

        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestDM(LightningDataModule):

        hook_called = False

        def transfer_batch_to_device(self, data, device):
            self.hook_called = True
            if isinstance(data, CustomBatch):
                data.samples = data.samples.to(device)
                data.targets = data.targets.to(device)
            else:
                data = super().transfer_batch_to_device(data, device)
            return data

    model = EvalModelTemplate()
    dm = CurrentTestDM()
    batch = CustomBatch((torch.zeros(5, 28), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer()
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    trainer.get_model = MagicMock(return_value=model)
    if trainer.is_overridden('transfer_batch_to_device', dm):
        model.transfer_batch_to_device = dm.transfer_batch_to_device

    batch_gpu = trainer.transfer_batch_to_gpu(batch, 0)
    expected = torch.device('cuda', 0)
    assert dm.hook_called
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected
