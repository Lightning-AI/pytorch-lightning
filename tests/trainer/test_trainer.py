import glob
import math
import os
import pickle
import types
from argparse import Namespace

import cloudpickle
import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml, save_hparams_to_tags_csv
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.utilities.io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


@pytest.mark.parametrize('url_ckpt', [True, False])
def test_no_val_module(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv('TORCH_HOME', tmpdir)

    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir)
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
    assert LightningModule.CHECKPOINT_KEY_HYPER_PARAMS in ckpt.keys(), 'module_arguments missing from checkpoints'

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, 'hparams.yaml')
    ckpt_path = f'http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}' if url_ckpt else new_weights_path
    model_2 = EvalModelTemplate.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file=hparams_path
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
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir)
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
        hparams_file=hparams_path
    )
    model_2.eval()


def test_gradient_accumulation_scheduling(tmpdir):
    """
    Test grad accumulation by the freq of optimizer updates
    """

    # test incorrect configs
    with pytest.raises(IndexError):
        assert Trainer(accumulate_grad_batches={0: 3, 1: 4, 4: 6})
        assert Trainer(accumulate_grad_batches={-2: 3})

    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches={})
        assert Trainer(accumulate_grad_batches=[[2, 3], [4, 6]])
        assert Trainer(accumulate_grad_batches={1: 2, 3.: 4})
        assert Trainer(accumulate_grad_batches={1: 2.5, 3: 5})

    # test optimizer call freq matches scheduler
    def _optimizer_step(self, epoch, batch_idx, optimizer,
                        optimizer_idx, second_order_closure=None):
        # only test the first 12 batches in epoch
        if batch_idx < 12:
            if epoch == 0:
                # reset counter when starting epoch
                if batch_idx == 0:
                    self.prev_called_batch_idx = 0

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 1

                assert batch_idx == self.prev_called_batch_idx
                self.prev_called_batch_idx += 1

            elif 1 <= epoch <= 2:
                # reset counter when starting epoch
                if batch_idx == 1:
                    self.prev_called_batch_idx = 1

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 2

                assert batch_idx == self.prev_called_batch_idx
                self.prev_called_batch_idx += 2

            else:
                if batch_idx == 3:
                    self.prev_called_batch_idx = 3

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 4

                assert batch_idx == self.prev_called_batch_idx
                self.prev_called_batch_idx += 3

        optimizer.step()

        # clear gradients
        optimizer.zero_grad()

    model = EvalModelTemplate()
    schedule = {1: 2, 3: 4}

    trainer = Trainer(accumulate_grad_batches=schedule,
                      train_percent_check=0.1,
                      val_percent_check=0.1,
                      max_epochs=2,
                      default_root_dir=tmpdir)

    # for the test
    trainer.optimizer_step = _optimizer_step
    model.prev_called_batch_idx = 0

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
        trainer.callback_metrics = {'val_loss': loss}
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
        max_epochs=1,
        checkpoint_callback=ModelCheckpoint(tmpdir, save_weights_only=True)
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

        def increment_batch(self, _):
            self.num_batches_seen += 1

        def increment_on_load_checkpoint(self, _):
            self.num_on_load_checkpoint_called += 1

        # Bind methods to keep track of epoch numbers, batch numbers it has seen
        # as well as number of times it has called on_load_checkpoint()
        model.on_epoch_end = types.MethodType(increment_epoch, model)
        model.on_batch_start = types.MethodType(increment_batch, model)
        model.on_load_checkpoint = types.MethodType(increment_on_load_checkpoint, model)
        return model

    model = _new_model()

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        train_percent_check=0.65,
        val_percent_check=1,
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
        train_percent_check=train_percent,
    )
    return model, trainer_options, num_train_samples


def test_trainer_max_steps_and_epochs(tmpdir):
    """Verify model trains according to specified max steps"""
    model, trainer_options, num_train_samples = _init_steps_model()

    # define less train steps than epochs
    trainer_options.update(
        default_root_dir=tmpdir,
        max_epochs=3,
        max_steps=num_train_samples + 10
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
        max_steps=trainer_options['max_epochs'] * 2 * num_train_samples
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
        max_epochs=2
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


def test_testpass_overrides(tmpdir):
    # todo: check duplicated tests against trainer_checks
    hparams = EvalModelTemplate.get_default_hparams()

    # Misconfig when neither test_step or test_end is implemented
    with pytest.raises(MisconfigurationException, match='.*not implement `test_dataloader`.*'):
        model = EvalModelTemplate(**hparams)
        model.test_dataloader = LightningModule.test_dataloader
        Trainer().test(model)

    # Misconfig when neither test_step or test_end is implemented
    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(**hparams)
        model.test_step = LightningModule.test_step
        Trainer().test(model)

    # No exceptions when one or both of test_step or test_end are implemented
    model = EvalModelTemplate(**hparams)
    model.test_step_end = LightningModule.test_step_end
    Trainer().test(model)

    model = EvalModelTemplate(**hparams)
    Trainer().test(model)


def test_disabled_validation():
    """Verify that `val_percent_check=0` disables the validation loop unless `fast_dev_run=True`."""

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
        progress_bar_refresh_rate=0,
        max_epochs=2,
        train_percent_check=0.4,
        val_percent_check=0.0,
        fast_dev_run=False,
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # check that val_percent_check=0 turns off validation
    assert result == 1, 'training failed to complete'
    assert trainer.current_epoch == 1
    assert not model.validation_step_invoked, \
        '`validation_step` should not run when `val_percent_check=0`'
    assert not model.validation_epoch_end_invoked, \
        '`validation_epoch_end` should not run when `val_percent_check=0`'

    # check that val_percent_check has no influence when fast_dev_run is turned on
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
        terminate_on_nan=True
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
        terminate_on_nan=True
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

        def on_batch_start(self, trainer, pl_module):
            raise KeyboardInterrupt

    interrupt_callback = InterruptCallback()

    trainer = Trainer(
        callbacks=[interrupt_callback],
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2,
        progress_bar_refresh_rate=0,
        logger=False,
        default_root_dir=tmpdir,
    )
    assert not trainer.interrupted
    trainer.fit(model)
    assert trainer.interrupted


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
        default_root_dir=tmpdir
    )

    # for the test
    model.optimizer_step = _optimizer_step
    model.prev_called_batch_idx = 0

    trainer.fit(model)


def test_gpu_choice(tmpdir):
    trainer_options = dict(
        default_save_path=tmpdir,
    )
    # Only run if CUDA is available
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()
    Trainer(**trainer_options, gpus=num_gpus, auto_select_gpus=True)

    with pytest.raises(RuntimeError, match=r'.*No GPUs available.*'):
        Trainer(**trainer_options, gpus=num_gpus + 1, auto_select_gpus=True)


@pytest.mark.parametrize("trainer_kwargs,expected", [
    pytest.param(
        dict(distributed_backend=None, gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="dp", gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="dp", gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="ddp", gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="ddp", num_processes=2, gpus=None),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=2)
    ),
    pytest.param(
        dict(distributed_backend="ddp", num_nodes=2, gpus=None),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend="ddp_cpu", num_processes=2, gpus=None),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=2)
    ),
    pytest.param(
        dict(distributed_backend="ddp2", gpus=None),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=1)
    ),
    pytest.param(
        dict(distributed_backend=None, gpus=1),
        dict(use_dp=False, use_ddp=False, use_ddp2=False, num_gpus=1, on_gpu=True, single_gpu=True, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend="dp", gpus=1),
        dict(use_dp=True, use_ddp=False, use_ddp2=False, num_gpus=1, on_gpu=True, single_gpu=True, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp", gpus=1),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=1, on_gpu=True, single_gpu=True, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp_cpu", num_processes=2, gpus=1),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=0, on_gpu=False, single_gpu=False, num_processes=2),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp2", gpus=1),
        dict(use_dp=False, use_ddp=False, use_ddp2=True, num_gpus=1, on_gpu=True, single_gpu=False, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() == 0, reason="GPU needed")]
    ),
    pytest.param(
        dict(distributed_backend=None, gpus=2),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=2, on_gpu=True, single_gpu=False, num_processes=2),
        marks=[pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs needed")]
    ),
    pytest.param(
        dict(distributed_backend="dp", gpus=2),
        dict(use_dp=True, use_ddp=False, use_ddp2=False, num_gpus=2, on_gpu=True, single_gpu=False, num_processes=1),
        marks=[pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp", gpus=2),
        dict(use_dp=False, use_ddp=True, use_ddp2=False, num_gpus=2, on_gpu=True, single_gpu=False, num_processes=2),
        marks=[pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs needed")]
    ),
    pytest.param(
        dict(distributed_backend="ddp2", gpus=2),
        dict(use_dp=False, use_ddp=False, use_ddp2=True, num_gpus=2, on_gpu=True, single_gpu=False, num_processes=1),
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
    assert trainer.single_gpu is expected["single_gpu"]
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


def test_trainer_pickle(tmpdir):
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir
    )
    pickle.dumps(trainer)
    cloudpickle.dumps(trainer)
