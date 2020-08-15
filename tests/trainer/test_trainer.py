import glob
import math
import os
import pickle
import sys
import types
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cloudpickle
import pytest
import torch
from omegaconf import OmegaConf

import tests.base.develop_utils as tutils
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.saving import (
    load_hparams_from_tags_csv, load_hparams_from_yaml, save_hparams_to_tags_csv)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_no_val_module(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', str(tmpdir))

    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir),
    )
    # fit model
    result = trainer.fit(model)
    # training complete
    assert result == 1, 'amp + ddp model failed to complete'

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # assert ckpt has hparams
    ckpt = torch.load(new_weights_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in ckpt.keys(), 'module_arguments missing from checkpoints'

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, 'hparams.yaml')
    ckpt_path = f'http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}' if url_ckpt else new_weights_path
    model_2 = EvalModelTemplate.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file=hparams_path,
    )
    model_2.eval()


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_no_val_end_module(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', tmpdir)

    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir),
    )
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, 'hparams.yaml')
    ckpt_path = f'http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}' if url_ckpt else new_weights_path
    model_2 = EvalModelTemplate.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file=hparams_path,
    )
    model_2.eval()


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_strict_model_load(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', tmpdir)

    model = EvalModelTemplate()
    # Extra layer
    model.c_d3 = torch.nn.Linear(model.hidden_dim, model.hidden_dim)

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir),
    )
    result = trainer.fit(model)

    # traning complete
    assert result == 1

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, 'hparams.yaml')
    ckpt_path = f'http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}' \
        if url_ckpt else new_weights_path

    try:
        EvalModelTemplate.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams_file=hparams_path,
        )
    except Exception:
        failed = True
    else:
        failed = False

    assert failed, "Model should not been loaded since the extra layer added."

    failed = False
    try:
        EvalModelTemplate.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams_file=hparams_path,
            strict=False,
        )
    except Exception:
        failed = True

    assert not failed, "Model should be loaded due to strict=False."


@pytest.mark.parametrize(
    ['schedule', 'expected'],
    [
        pytest.param({1: 2, 3: 4}, [1, 2, 4]),
        pytest.param(3, [3, 3, 3]),
        pytest.param(4, [4, 4, 4])
    ]
)
def test_gradient_accumulation_scheduling(tmpdir, schedule, expected):
    """
    Test grad accumulation by the freq of optimizer updates
    """

    # test incorrect configs
    with pytest.raises(IndexError):
        assert Trainer(accumulate_grad_batches={-1: 3, 1: 4, 4: 6})
    with pytest.raises(IndexError):
        assert Trainer(accumulate_grad_batches={-2: 3})

    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches={})
    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches=[[2, 3], [4, 6]])
    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches={1: 2, 3.: 4})
    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches={1: 2.5, 3: 5})

    model = EvalModelTemplate()

    trainer = Trainer(
        accumulate_grad_batches=schedule,
        limit_train_batches=0.7,  # not to be divisible by accumulate_grad_batches on purpose
        limit_val_batches=0.8,
        max_epochs=4,
        default_root_dir=tmpdir,
    )

    # test optimizer call freq matches scheduler
    def _optimizer_step(epoch, batch_idx, optimizer, optimizer_idx,
                        second_order_closure=None, on_tpu=False,
                        using_native_amp=False, using_lbfgs=False):
        # only test the first 12 batches in epoch
        if batch_idx < 12:
            if epoch == 0:
                # reset counter when starting epoch
                if batch_idx == expected[0] - 1:
                    model.prev_called_batch_idx = expected[0] - 1

                    # use this opportunity to test once
                    assert trainer.accumulate_grad_batches == expected[0]

                # separate check for last batch with accumulate 1 step
                if expected[0] == 1 and (batch_idx + 1) == trainer.num_training_batches:
                    assert batch_idx == model.prev_called_batch_idx
                elif (batch_idx + 1) == trainer.num_training_batches:
                    # prev_called_batch_idx - schedule + modulus remainder
                    assert batch_idx == (model.prev_called_batch_idx - expected[0] + (batch_idx + 1) % expected[0])
                else:
                    assert batch_idx == model.prev_called_batch_idx
                    model.prev_called_batch_idx += expected[0]

            elif 1 <= epoch <= 2:
                # reset counter when starting epoch
                if batch_idx == expected[1] - 1:
                    model.prev_called_batch_idx = expected[1] - 1

                    # use this opportunity to test once
                    assert trainer.accumulate_grad_batches == expected[1]

                if trainer.num_training_batches == batch_idx + 1:
                    # prev_called_batch_idx - schedule + modulus remainder
                    assert batch_idx == (model.prev_called_batch_idx - expected[1] + (batch_idx + 1) % expected[1])
                else:
                    assert batch_idx == model.prev_called_batch_idx
                    model.prev_called_batch_idx += expected[1]

            else:
                if batch_idx == expected[2] - 1:
                    model.prev_called_batch_idx = expected[2] - 1

                    # use this opportunity to test once
                    assert trainer.accumulate_grad_batches == expected[2]

                if (batch_idx + 1) == trainer.num_training_batches:
                    # prev_called_batch_idx - schedule + modulus remainder
                    assert batch_idx == (model.prev_called_batch_idx - expected[2] + (batch_idx + 1) % expected[2])
                else:
                    assert batch_idx == model.prev_called_batch_idx
                    model.prev_called_batch_idx += expected[2]

        optimizer.step()

        # clear gradients
        optimizer.zero_grad()

    # for the test
    model.optimizer_step = _optimizer_step
    model.prev_called_batch_idx = 0

    trainer.fit(model)


@pytest.mark.parametrize(
    ['accumulate_grad_batches', 'limit_train_batches'],
    [
        pytest.param({1: 2, 3: 4}, 1.0),
        pytest.param({1: 2, 3: 4}, 0.5),  # not to be divisible by accumulate_grad_batches on purpose
        pytest.param(3, 1.0),
        pytest.param(3, 0.8),  # not to be divisible by accumulate_grad_batches on purpose
        pytest.param(4, 1.0),
        pytest.param(4, 0.7),  # not to be divisible by accumulate_grad_batches on purpose
    ],
)
def test_gradient_accumulation_scheduling_last_batch(tmpdir, accumulate_grad_batches, limit_train_batches):
    """ Verify optimizer.step() applied to last batch while grad accumulation """

    class CurrentModel(EvalModelTemplate):
        def on_after_backward(self):
            self.loss_backward = deepcopy(self.state_dict())

        def on_before_zero_grad(self, optimizer):
            self.opt_step = self.state_dict()

        def on_train_batch_end(self, batch, batch_idx, dataloader_idx):
            _exclude_keys = ['num_batches_tracked', 'running_mean', 'running_var']

            if (batch_idx + 1) == self.trainer.num_training_batches:
                for key in self.loss_backward.keys():
                    # exclude the check for batch_norm parameters
                    if not any([k in key for k in _exclude_keys]):
                        assert not torch.equal(self.loss_backward[key], self.opt_step[key])

    model = CurrentModel()

    trainer = Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=4,
        limit_train_batches=limit_train_batches,
        default_root_dir=tmpdir
    )

    trainer.fit(model)


def test_loading_meta_tags(tmpdir):
    """ test for backward compatibility to meta_tags.csv """
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()

    # save tags
    logger = tutils.get_default_logger(tmpdir)
    logger.log_hyperparams(Namespace(some_str='a_str', an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load hparams
    path_expt_dir = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(path_expt_dir, TensorBoardLogger.NAME_HPARAMS_FILE)
    hparams = load_hparams_from_yaml(hparams_path)

    # save as legacy meta_tags.csv
    tags_path = os.path.join(path_expt_dir, 'meta_tags.csv')
    save_hparams_to_tags_csv(tags_path, hparams)

    tags = load_hparams_from_tags_csv(tags_path)

    assert hparams == tags


def test_loading_yaml(tmpdir):
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()

    # save tags
    logger = tutils.get_default_logger(tmpdir)
    logger.log_hyperparams(Namespace(some_str='a_str', an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load hparams
    path_expt_dir = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(path_expt_dir, 'hparams.yaml')
    tags = load_hparams_from_yaml(hparams_path)

    assert tags['batch_size'] == 32 and tags['hidden_dim'] == 1000


def test_dp_output_reduce():
    mixin = TrainerLoggingMixin()

    # test identity when we have a single gpu
    out = torch.rand(3, 1)
    assert mixin.reduce_distributed_output(out, num_gpus=1) is out

    # average when we have multiples
    assert mixin.reduce_distributed_output(out, num_gpus=2) == out.mean()

    # when we have a dict of vals
    out = {
        'a': out,
        'b': {
            'c': out
        }
    }
    reduced = mixin.reduce_distributed_output(out, num_gpus=3)
    assert reduced['a'] == out['a']
    assert reduced['b']['c'] == out['b']['c']


@pytest.mark.parametrize(["save_top_k", "save_last", "file_prefix", "expected_files"], [
    pytest.param(-1, False, '', {'epoch=4.ckpt', 'epoch=3.ckpt', 'epoch=2.ckpt', 'epoch=1.ckpt', 'epoch=0.ckpt'},
                 id="CASE K=-1  (all)"),
    pytest.param(1, False, 'test_prefix_', {'test_prefix_epoch=4.ckpt'},
                 id="CASE K=1 (2.5, epoch 4)"),
    pytest.param(2, False, '', {'epoch=4.ckpt', 'epoch=2.ckpt'},
                 id="CASE K=2 (2.5 epoch 4, 2.8 epoch 2)"),
    pytest.param(4, False, '', {'epoch=1.ckpt', 'epoch=4.ckpt', 'epoch=3.ckpt', 'epoch=2.ckpt'},
                 id="CASE K=4 (save all 4 base)"),
    pytest.param(3, False, '', {'epoch=2.ckpt', 'epoch=3.ckpt', 'epoch=4.ckpt'},
                 id="CASE K=3 (save the 2nd, 3rd, 4th model)"),
    pytest.param(1, True, '', {'epoch=4.ckpt', 'last.ckpt'},
                 id="CASE K=1 (save the 4th model and the last model)"),
])
def test_model_checkpoint_options(tmpdir, save_top_k, save_last, file_prefix, expected_files):
    """Test ModelCheckpoint options."""

    def mock_save_function(filepath, *args):
        open(filepath, 'a').close()

    # simulated losses
    losses = [10, 9, 2.8, 5, 2.5]

    checkpoint_callback = ModelCheckpoint(tmpdir, save_top_k=save_top_k, save_last=save_last,
                                          prefix=file_prefix, verbose=1)
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for i, loss in enumerate(losses):
        trainer.current_epoch = i
        trainer.callback_metrics = {'val_loss': torch.tensor(loss)}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

    file_lists = set(os.listdir(tmpdir))

    assert len(file_lists) == len(expected_files), \
        "Should save %i models when save_top_k=%i" % (len(expected_files), save_top_k)

    # verify correct naming
    for fname in expected_files:
        assert fname in file_lists


def test_model_checkpoint_only_weights(tmpdir):
    """Tests use case where ModelCheckpoint is configured to save only model weights, and
     user tries to load checkpoint to resume training.
     """
    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        checkpoint_callback=ModelCheckpoint(tmpdir, save_weights_only=True),
    )
    # fit model
    result = trainer.fit(model)
    # training complete
    assert result == 1, 'training failed to complete'

    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]

    # assert saved checkpoint has no trainer data
    checkpoint = torch.load(checkpoint_path)
    assert 'optimizer_states' not in checkpoint, 'checkpoint should contain only model weights'
    assert 'lr_schedulers' not in checkpoint, 'checkpoint should contain only model weights'

    # assert loading model works when checkpoint has only weights
    assert EvalModelTemplate.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # directly save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path, weights_only=True)
    # assert saved checkpoint has no trainer data
    checkpoint = torch.load(new_weights_path)
    assert 'optimizer_states' not in checkpoint, 'checkpoint should contain only model weights'
    assert 'lr_schedulers' not in checkpoint, 'checkpoint should contain only model weights'

    # assert restoring train state fails
    with pytest.raises(KeyError, match='checkpoint contains only the model'):
        trainer.restore_training_state(checkpoint)


def test_model_freeze_unfreeze():

    model = EvalModelTemplate()

    model.freeze()
    model.unfreeze()


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_resume_from_checkpoint_epoch_restored(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Verify resuming from checkpoint runs the right number of epochs"""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', tmpdir)

    hparams = EvalModelTemplate.get_default_hparams()

    def _new_model():
        # Create a model that tracks epochs and batches seen
        model = EvalModelTemplate(**hparams)
        model.num_epochs_seen = 0
        model.num_batches_seen = 0
        model.num_on_load_checkpoint_called = 0

        def increment_epoch(self):
            self.num_epochs_seen += 1

        def increment_batch(self, batch, batch_idx, dataloader_idx):
            self.num_batches_seen += 1

        def increment_on_load_checkpoint(self, _):
            self.num_on_load_checkpoint_called += 1

        # Bind methods to keep track of epoch numbers, batch numbers it has seen
        # as well as number of times it has called on_load_checkpoint()
        model.on_epoch_end = types.MethodType(increment_epoch, model)
        model.on_train_batch_start = types.MethodType(increment_batch, model)
        model.on_load_checkpoint = types.MethodType(increment_on_load_checkpoint, model)
        return model

    model = _new_model()

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=0.65,
        limit_val_batches=1,
        checkpoint_callback=ModelCheckpoint(tmpdir, save_top_k=-1),
        default_root_dir=tmpdir,
        early_stop_callback=False,
        val_check_interval=1.,
    )

    trainer = Trainer(**trainer_options)
    # fit model
    trainer.fit(model)

    training_batches = trainer.num_training_batches

    assert model.num_epochs_seen == 2
    assert model.num_batches_seen == training_batches * 2
    assert model.num_on_load_checkpoint_called == 0

    # Other checkpoints can be uncommented if/when resuming mid-epoch is supported
    checkpoints = sorted(glob.glob(os.path.join(trainer.checkpoint_callback.dirpath, '*.ckpt')))
    if url_ckpt:
        # transform local paths into url checkpoints
        ip, port = tmpdir_server
        checkpoints = [f'http://{ip}:{port}/' + os.path.basename(check) for check in checkpoints]

    for check in checkpoints:
        next_model = _new_model()
        state = pl_load(check)

        # Resume training
        trainer_options['max_epochs'] = 2
        new_trainer = Trainer(**trainer_options, resume_from_checkpoint=check)
        new_trainer.fit(next_model)
        assert state['global_step'] + next_model.num_batches_seen == training_batches * trainer_options['max_epochs']
        assert next_model.num_on_load_checkpoint_called == 1


def _init_steps_model():
    """private method for initializing a model with 5% train epochs"""
    model = EvalModelTemplate()

    # define train epoch to 5% of data
    train_percent = 0.5
    # get number of samples in 1 epoch
    num_train_samples = math.floor(len(model.train_dataloader()) * train_percent)

    trainer_options = dict(
        limit_train_batches=train_percent,
    )
    return model, trainer_options, num_train_samples


def test_trainer_max_steps_and_epochs(tmpdir):
    """Verify model trains according to specified max steps"""
    model, trainer_options, num_train_samples = _init_steps_model()

    # define less train steps than epochs
    trainer_options.update(
        default_root_dir=tmpdir,
        max_epochs=3,
        max_steps=num_train_samples + 10,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result == 1, "Training did not complete"

    # check training stopped at max_steps
    assert trainer.global_step == trainer.max_steps, "Model did not stop at max_steps"

    # define less train epochs than steps
    trainer_options.update(
        max_epochs=2,
        max_steps=trainer_options['max_epochs'] * 2 * num_train_samples,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result == 1, "Training did not complete"

    # check training stopped at max_epochs
    assert trainer.global_step == num_train_samples * trainer.max_epochs
    assert trainer.current_epoch == trainer.max_epochs - 1, "Model did not stop at max_epochs"


def test_trainer_min_steps_and_epochs(tmpdir):
    """Verify model trains according to specified min steps"""
    model, trainer_options, num_train_samples = _init_steps_model()

    # define callback for stopping the model and default epochs
    trainer_options.update(
        default_root_dir=tmpdir,
        early_stop_callback=EarlyStopping(monitor='val_loss', min_delta=1.0),
        val_check_interval=2,
        min_epochs=1,
        max_epochs=7,
    )

    # define less min steps than 1 epoch
    trainer_options['min_steps'] = math.floor(num_train_samples / 2)

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result == 1, "Training did not complete"

    # check model ran for at least min_epochs
    assert trainer.global_step >= num_train_samples and \
        trainer.current_epoch > 0, "Model did not train for at least min_epochs"

    # define less epochs than min_steps
    trainer_options['min_steps'] = math.floor(num_train_samples * 1.5)

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result == 1, "Training did not complete"

    # check model ran for at least num_train_samples*1.5
    assert trainer.global_step >= math.floor(num_train_samples * 1.5) and \
        trainer.current_epoch > 0, "Model did not train for at least min_steps"


def test_benchmark_option(tmpdir):
    """Verify benchmark option."""

    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__multiple

    # verify torch.backends.cudnn.benchmark is not turned on
    assert not torch.backends.cudnn.benchmark

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        benchmark=True,
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1

    # verify torch.backends.cudnn.benchmark is not turned off
    assert torch.backends.cudnn.benchmark


@pytest.mark.parametrize('ckpt_path', [None, 'best', 'specific'])
@pytest.mark.parametrize('save_top_k', [-1, 0, 1, 2])
def test_test_checkpoint_path(tmpdir, ckpt_path, save_top_k):
    hparams = EvalModelTemplate.get_default_hparams()

    model = EvalModelTemplate(**hparams)
    trainer = Trainer(
        max_epochs=2,
        progress_bar_refresh_rate=0,
        default_root_dir=tmpdir,
        checkpoint_callback=ModelCheckpoint(save_top_k=save_top_k),
    )
    trainer.fit(model)
    if ckpt_path == 'best':
        # ckpt_path is 'best', meaning we load the best weights
        if save_top_k <= 0:
            with pytest.raises(MisconfigurationException, match='.*is not configured to save the best.*'):
                trainer.test(ckpt_path=ckpt_path)
        else:
            trainer.test(ckpt_path=ckpt_path)
            assert trainer.tested_ckpt_path == trainer.checkpoint_callback.best_model_path
    elif ckpt_path is None:
        # ckpt_path is None, meaning we don't load any checkpoints and
        # use the weights from the end of training
        trainer.test(ckpt_path=ckpt_path)
        assert trainer.tested_ckpt_path is None
    else:
        # specific checkpoint, pick one from saved ones
        if save_top_k == 0:
            with pytest.raises(FileNotFoundError):
                trainer.test(ckpt_path='random.ckpt')
        else:
            ckpt_path = str(list((Path(tmpdir) / f'lightning_logs/version_{trainer.logger.version}/checkpoints').iterdir())[0].absolute())
            trainer.test(ckpt_path=ckpt_path)
            assert trainer.tested_ckpt_path == ckpt_path


def test_disabled_validation(tmpdir):
    """Verify that `limit_val_batches=0` disables the validation loop unless `fast_dev_run=True`."""

    class CurrentModel(EvalModelTemplate):

        validation_step_invoked = False
        validation_epoch_end_invoked = False

        def validation_step(self, *args, **kwargs):
            self.validation_step_invoked = True
            return super().validation_step(*args, **kwargs)

        def validation_epoch_end(self, *args, **kwargs):
            self.validation_epoch_end_invoked = True
            return super().validation_epoch_end(*args, **kwargs)

    hparams = EvalModelTemplate.get_default_hparams()
    model = CurrentModel(**hparams)

    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.0,
        fast_dev_run=False,
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # check that limit_val_batches=0 turns off validation
    assert result == 1, 'training failed to complete'
    assert trainer.current_epoch == 1
    assert not model.validation_step_invoked, \
        '`validation_step` should not run when `limit_val_batches=0`'
    assert not model.validation_epoch_end_invoked, \
        '`validation_epoch_end` should not run when `limit_val_batches=0`'

    # check that limit_val_batches has no influence when fast_dev_run is turned on
    model = CurrentModel(**hparams)
    trainer_options.update(fast_dev_run=True)
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'
    assert trainer.current_epoch == 0
    assert model.validation_step_invoked, \
        'did not run `validation_step` with `fast_dev_run=True`'
    assert model.validation_epoch_end_invoked, \
        'did not run `validation_epoch_end` with `fast_dev_run=True`'


def test_nan_loss_detection(tmpdir):

    class CurrentModel(EvalModelTemplate):
        test_batch_inf_loss = 8

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            output = super().training_step(batch, batch_idx, optimizer_idx)
            if batch_idx == self.test_batch_inf_loss:
                if isinstance(output, dict):
                    output['loss'] *= torch.tensor(math.inf)  # make loss infinite
                else:
                    output /= 0
            return output

    model = CurrentModel()

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=(model.test_batch_inf_loss + 1),
        terminate_on_nan=True,
    )

    with pytest.raises(ValueError, match=r'.*The loss returned in `training_step` is nan or inf.*'):
        trainer.fit(model)
        assert trainer.global_step == model.test_step_inf_loss

    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_nan_params_detection(tmpdir):

    class CurrentModel(EvalModelTemplate):
        test_batch_nan = 8

        def on_after_backward(self):
            if self.global_step == self.test_batch_nan:
                # simulate parameter that became nan
                torch.nn.init.constant_(self.c_d1.bias, math.nan)

    model = CurrentModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=(model.test_batch_nan + 1),
        terminate_on_nan=True,
    )

    with pytest.raises(ValueError, match=r'.*Detected nan and/or inf values in `c_d1.bias`.*'):
        trainer.fit(model)
        assert trainer.global_step == model.test_batch_nan

    # after aborting the training loop, model still has nan-valued params
    params = torch.cat([param.view(-1) for param in model.parameters()])
    assert not torch.isfinite(params).all()


def test_trainer_interrupted_flag(tmpdir):
    """Test the flag denoting that a user interrupted training."""

    model = EvalModelTemplate()

    class InterruptCallback(Callback):
        def __init__(self):
            super().__init__()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            raise KeyboardInterrupt

    class HandleInterruptCallback(Callback):
        def __init__(self):
            super().__init__()
            self.exc_info = None

        def on_keyboard_interrupt(self, trainer, pl_module):
            self.exc_info = sys.exc_info()

    interrupt_callback = InterruptCallback()
    handle_interrupt_callback = HandleInterruptCallback()

    trainer = Trainer(
        callbacks=[interrupt_callback, handle_interrupt_callback],
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        progress_bar_refresh_rate=0,
        logger=False,
        default_root_dir=tmpdir,
    )
    assert not trainer.interrupted
    assert handle_interrupt_callback.exc_info is None
    trainer.fit(model)
    assert trainer.interrupted
    assert isinstance(handle_interrupt_callback.exc_info[1], KeyboardInterrupt)


def test_gradient_clipping(tmpdir):
    """
    Test gradient clipping
    """

    model = EvalModelTemplate()

    # test that gradient is clipped correctly
    def _optimizer_step(*args, **kwargs):
        parameters = model.parameters()
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
        assert (grad_norm - 1.0).abs() < 0.01, "Gradient norm != 1.0: {grad_norm}".format(grad_norm=grad_norm)

    trainer = Trainer(
        max_steps=1,
        max_epochs=1,
        gradient_clip_val=1.0,
        default_root_dir=tmpdir,
    )

    # for the test
    model.optimizer_step = _optimizer_step
    model.prev_called_batch_idx = 0

    trainer.fit(model)


def test_gpu_choice(tmpdir):
    trainer_options = dict(
        default_root_dir=tmpdir,
    )
    # Only run if CUDA is available
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()
    Trainer(**trainer_options, gpus=num_gpus, auto_select_gpus=True)

    with pytest.raises(RuntimeError, match=r'.*No GPUs available.*'):
        Trainer(**trainer_options, gpus=num_gpus + 1, auto_select_gpus=True)


@pytest.mark.parametrize(['tpu_cores', 'expected_tpu_id', 'error_expected'], [
    pytest.param(1, None, False),
    pytest.param(8, None, False),
    pytest.param([1], 1, False),
    pytest.param([8], 8, False),
    pytest.param('1,', 1, False),
    pytest.param('1', None, False),
    pytest.param('9, ', 9, True),
    pytest.param([9], 9, True),
    pytest.param([0], 0, True),
    pytest.param(2, None, True),
    pytest.param(10, None, True),
])
def test_tpu_choice(tmpdir, tpu_cores, expected_tpu_id, error_expected):
    if error_expected:
        with pytest.raises(MisconfigurationException, match=r'.*tpu_cores` can only be 1, 8 or [<1-8>]*'):
            Trainer(default_root_dir=tmpdir, tpu_cores=tpu_cores, auto_select_gpus=True)
    else:
        trainer = Trainer(default_root_dir=tmpdir, tpu_cores=tpu_cores, auto_select_gpus=True)
        assert trainer.tpu_id == expected_tpu_id


@pytest.mark.parametrize(['limit_val_batches'], [
    pytest.param(0.0),  # this should run no sanity checks
    pytest.param(1),
    pytest.param(1.0),
    pytest.param(0.3),
])
def test_num_sanity_val_steps(tmpdir, limit_val_batches):
    """
    Test that num_sanity_val_steps=-1 runs through all validation data once.
    Makes sure this setting is independent of limit_val_batches.
    """
    model = EvalModelTemplate()
    model.validation_step = model.validation_step__multiple_dataloaders
    model.validation_epoch_end = model.validation_epoch_end__multiple_dataloaders
    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=-1,
        limit_val_batches=limit_val_batches,  # should have no influence
        max_steps=1,
    )
    assert trainer.num_sanity_val_steps == float('inf')
    val_dataloaders = model.val_dataloader__multiple()

    with patch.object(trainer, 'evaluation_forward', wraps=trainer.evaluation_forward) as mocked:
        trainer.fit(model, val_dataloaders=val_dataloaders)
        assert mocked.call_count == sum(len(dl) * (limit_val_batches > 0) for dl in val_dataloaders)


@pytest.mark.parametrize("trainer_kwargs,expected", [
    pytest.param(
        dict(distributed_backend=None, gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="dp", gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="dp", gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="ddp", gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="ddp", num_processes=2, gpus=None),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=2)
    ),
    pytest.param(
        dict(distributed_backend="ddp", num_nodes=2, gpus=None),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="ddp_cpu", num_processes=2, gpus=None),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=2)
    ),
    pytest.param(
        dict(distributed_backend="ddp2", gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend=None, gpus=1),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=1, on_gpu=True, use_single_gpu=True, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend="dp", gpus=1),
        dict(use_dp=True, use_ddp=False, use_ddp2=False, num_gpus=1, on_gpu=True, use_single_gpu=True, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp", gpus=1),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=1, on_gpu=True, use_single_gpu=True, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp_cpu", num_processes=2, gpus=1),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=0, on_gpu=False, use_single_gpu=False, num_processes=2),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp2", gpus=1),
        dict(use_dp=False, use_ddp=False, use_ddp2=True, num_gpus=1, on_gpu=True, use_single_gpu=False, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend=None, gpus=2),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=2, on_gpu=True, use_single_gpu=False, num_processes=2),
        marks=[pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs needed")]
    ),
    pytest.param(
        dict(distributed_backend="dp", gpus=2),
        dict(use_dp=True, use_ddp=False, use_ddp2=False, num_gpus=2, on_gpu=True, use_single_gpu=False, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp", gpus=2),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=2, on_gpu=True, use_single_gpu=False, num_processes=2),
        marks=[pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp2", gpus=2),
        dict(use_dp=False, use_ddp=False, use_ddp2=True, num_gpus=2, on_gpu=True, use_single_gpu=False, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs needed")]
    ),
])
def test_trainer_config(trainer_kwargs, expected):
    trainer = Trainer(**trainer_kwargs)
    assert trainer.use_dp is expected["use_dp"]
    assert trainer.use_ddp is expected["use_ddp"]
    assert trainer.use_ddp2 is expected["use_ddp2"]
    assert trainer.num_gpus == expected["num_gpus"]
    assert trainer.on_gpu is expected["on_gpu"]
    assert trainer.use_single_gpu is expected["use_single_gpu"]
    assert trainer.num_processes == expected["num_processes"]


def test_trainer_subclassing():
    model = EvalModelTemplate()

    # First way of pulling out args from signature is to list them
    class TrainerSubclass(Trainer):

        def __init__(self, custom_arg, *args, custom_kwarg='test', **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_arg = custom_arg
            self.custom_kwarg = custom_kwarg

    trainer = TrainerSubclass(123, custom_kwarg='custom', fast_dev_run=True)
    result = trainer.fit(model)
    assert result == 1
    assert trainer.custom_arg == 123
    assert trainer.custom_kwarg == 'custom'
    assert trainer.fast_dev_run

    # Second way is to pop from the dict
    # It's a special case because Trainer does not have any positional args
    class TrainerSubclass(Trainer):

        def __init__(self, **kwargs):
            self.custom_arg = kwargs.pop('custom_arg', 0)
            self.custom_kwarg = kwargs.pop('custom_kwarg', 'test')
            super().__init__(**kwargs)

    trainer = TrainerSubclass(custom_kwarg='custom', fast_dev_run=True)
    result = trainer.fit(model)
    assert result == 1
    assert trainer.custom_kwarg == 'custom'
    assert trainer.fast_dev_run

    # when we pass in an unknown arg, the base class should complain
    with pytest.raises(TypeError, match=r"__init__\(\) got an unexpected keyword argument 'abcdefg'"):
        TrainerSubclass(abcdefg='unknown_arg')


@pytest.mark.parametrize('trainer_params', [
    OmegaConf.create({'max_epochs': 1, 'gpus': 1}),
    OmegaConf.create({'max_epochs': 1, 'gpus': [0]}),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_trainer_omegaconf(trainer_params):
    Trainer(**trainer_params)


def test_trainer_pickle(tmpdir):
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
    )
    pickle.dumps(trainer)
    cloudpickle.dumps(trainer)


def test_trainer_setup_call(tmpdir):
    """Test setup call with fit and test call."""

    class CurrentModel(EvalModelTemplate):

        def setup(self, stage):
            self.stage = stage

    class TrainerSubclass(Trainer):

        def setup(self, stage):
            self.stage = stage

    model = CurrentModel()

    # fit model
    trainer = TrainerSubclass(
        default_root_dir=tmpdir,
        max_epochs=1,
        checkpoint_callback=False
    )

    trainer.fit(model)
    assert trainer.stage == 'fit'
    assert trainer.get_model().stage == 'fit'

    trainer.test(ckpt_path=None)
    assert trainer.stage == 'test'
    assert trainer.get_model().stage == 'test'
