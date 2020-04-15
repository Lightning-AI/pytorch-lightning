import glob
import math
import os
from argparse import Namespace, ArgumentParser

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Callback
from pytorch_lightning.core.lightning import load_hparams_from_tags_csv
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import (
    TestModelBase,
    DictHparamsModel,
    LightningTestModel,
    LightEmptyTestStep,
    LightValidationStepMixin,
    LightValidationMultipleDataloadersMixin,
    LightTrainDataloader,
    LightTestDataloader,
    LightValidationMixin,
)


def test_hparams_save_load(tmpdir):
    model = DictHparamsModel({'in_features': 28 * 28, 'out_features': 10})

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1

    # try to load the model now
    pretrained_model = tutils.load_model_from_checkpoint(
        trainer.checkpoint_callback.dirpath,
        module_class=DictHparamsModel
    )


def test_no_val_module(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()

    class CurrentTestModel(LightTrainDataloader, TestModelBase):
        pass

    model = CurrentTestModel(hparams)

    # logger file to get meta
    logger = tutils.get_default_testtube_logger(tmpdir, False)

    trainer_options = dict(
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # training complete
    assert result == 1, 'amp + ddp model failed to complete'

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    tags_path = tutils.get_data_path(logger, path_dir=tmpdir)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_checkpoint(
        checkpoint_path=new_weights_path,
        tags_csv=tags_path
    )
    model_2.eval()


def test_no_val_end_module(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    tutils.reset_seed()

    class CurrentTestModel(LightTrainDataloader, LightValidationStepMixin, TestModelBase):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    logger = tutils.get_default_testtube_logger(tmpdir, False)

    trainer_options = dict(
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    tags_path = tutils.get_data_path(logger, path_dir=tmpdir)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_checkpoint(
        checkpoint_path=new_weights_path,
        tags_csv=tags_path
    )
    model_2.eval()


def test_gradient_accumulation_scheduling(tmpdir):
    """
    Test grad accumulation by the freq of optimizer updates
    """
    tutils.reset_seed()

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

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)
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
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()

    # save tags
    logger = tutils.get_default_testtube_logger(tmpdir, False)
    logger.log_hyperparams(Namespace(some_str='a_str', an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load tags
    path_expt_dir = tutils.get_data_path(logger, path_dir=tmpdir)
    tags_path = os.path.join(path_expt_dir, 'meta_tags.csv')
    tags = load_hparams_from_tags_csv(tags_path)

    assert tags.batch_size == 32 and tags.hidden_dim == 1000


def test_dp_output_reduce():
    mixin = TrainerLoggingMixin()
    tutils.reset_seed()

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


@pytest.mark.parametrize(["save_top_k", "file_prefix", "expected_files"], [
    pytest.param(-1, '', {'epoch=4.ckpt', 'epoch=3.ckpt', 'epoch=2.ckpt', 'epoch=1.ckpt', 'epoch=0.ckpt'},
                 id="CASE K=-1  (all)"),
    pytest.param(1, 'test_prefix_', {'test_prefix_epoch=4.ckpt'},
                 id="CASE K=1 (2.5, epoch 4)"),
    pytest.param(2, '', {'epoch=4.ckpt', 'epoch=2.ckpt'},
                 id="CASE K=2 (2.5 epoch 4, 2.8 epoch 2)"),
    pytest.param(4, '', {'epoch=1.ckpt', 'epoch=4.ckpt', 'epoch=3.ckpt', 'epoch=2.ckpt'},
                 id="CASE K=4 (save all 4 base)"),
    pytest.param(3, '', {'epoch=2.ckpt', 'epoch=3.ckpt', 'epoch=4.ckpt'},
                 id="CASE K=3 (save the 2nd, 3rd, 4th model)"),
])
def test_model_checkpoint_options(tmpdir, save_top_k, file_prefix, expected_files):
    """Test ModelCheckpoint options."""

    def mock_save_function(filepath):
        open(filepath, 'a').close()

    hparams = tutils.get_default_hparams()
    _ = LightningTestModel(hparams)

    # simulated losses
    losses = [10, 9, 2.8, 5, 2.5]

    checkpoint_callback = ModelCheckpoint(tmpdir, save_top_k=save_top_k, prefix=file_prefix, verbose=1)
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


def test_model_freeze_unfreeze():
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    model.freeze()
    model.unfreeze()


def test_resume_from_checkpoint_epoch_restored(tmpdir):
    """Verify resuming from checkpoint runs the right number of epochs"""
    import types

    tutils.reset_seed()

    hparams = tutils.get_default_hparams()

    def _new_model():
        # Create a model that tracks epochs and batches seen
        model = LightningTestModel(hparams)
        model.num_epochs_seen = 0
        model.num_batches_seen = 0

        def increment_epoch(self):
            self.num_epochs_seen += 1

        def increment_batch(self, _):
            self.num_batches_seen += 1

        # Bind the increment_epoch function on_epoch_end so that the
        # model keeps track of the number of epochs it has seen.
        model.on_epoch_end = types.MethodType(increment_epoch, model)
        model.on_batch_start = types.MethodType(increment_batch, model)
        return model

    model = _new_model()

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        train_percent_check=0.65,
        val_percent_check=1,
        checkpoint_callback=ModelCheckpoint(tmpdir, save_top_k=-1),
        logger=False,
        default_root_dir=tmpdir,
        early_stop_callback=False,
        val_check_interval=1.,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    training_batches = trainer.num_training_batches

    assert model.num_epochs_seen == 2
    assert model.num_batches_seen == training_batches * 2

    # Other checkpoints can be uncommented if/when resuming mid-epoch is supported
    checkpoints = sorted(glob.glob(os.path.join(trainer.checkpoint_callback.dirpath, '*.ckpt')))

    for check in checkpoints:
        next_model = _new_model()
        state = torch.load(check)

        # Resume training
        trainer_options['max_epochs'] = 2
        new_trainer = Trainer(**trainer_options, resume_from_checkpoint=check)
        new_trainer.fit(next_model)
        assert state['global_step'] + next_model.num_batches_seen == training_batches * trainer_options['max_epochs']


def _init_steps_model():
    """private method for initializing a model with 5% train epochs"""
    tutils.reset_seed()
    model, _ = tutils.get_default_model()

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
    trainer_options.update(dict(
        default_root_dir=tmpdir,
        max_epochs=3,
        max_steps=num_train_samples + 10
    ))

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result == 1, "Training did not complete"

    # check training stopped at max_steps
    assert trainer.global_step == trainer.max_steps, "Model did not stop at max_steps"

    # define less train epochs than steps
    trainer_options.update(dict(
        max_epochs=2,
        max_steps=trainer_options['max_epochs'] * 2 * num_train_samples
    ))

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
    trainer_options.update(dict(
        default_root_dir=tmpdir,
        early_stop_callback=EarlyStopping(monitor='val_loss', min_delta=1.0),
        val_check_interval=2,
        min_epochs=1,
        max_epochs=5
    ))

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
    tutils.reset_seed()

    class CurrentTestModel(
        LightValidationMultipleDataloadersMixin,
        LightTrainDataloader,
        TestModelBase
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # verify torch.backends.cudnn.benchmark is not turned on
    assert not torch.backends.cudnn.benchmark

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        benchmark=True,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # verify training completed
    assert result == 1

    # verify torch.backends.cudnn.benchmark is not turned off
    assert torch.backends.cudnn.benchmark


def test_testpass_overrides(tmpdir):
    hparams = tutils.get_default_hparams()

    class LocalModel(LightTrainDataloader, TestModelBase):
        pass

    class LocalModelNoEnd(LightTrainDataloader, LightTestDataloader, LightEmptyTestStep, TestModelBase):
        pass

    class LocalModelNoStep(LightTrainDataloader, TestModelBase):
        def test_epoch_end(self, outputs):
            return {}

    # Misconfig when neither test_step or test_end is implemented
    with pytest.raises(MisconfigurationException):
        model = LocalModel(hparams)
        Trainer().test(model)

    # Misconfig when neither test_step or test_end is implemented
    with pytest.raises(MisconfigurationException):
        model = LocalModelNoStep(hparams)
        Trainer().test(model)

    # No exceptions when one or both of test_step or test_end are implemented
    model = LocalModelNoEnd(hparams)
    Trainer().test(model)

    model = LightningTestModel(hparams)
    Trainer().test(model)


def test_disabled_validation():
    """Verify that `val_percent_check=0` disables the validation loop unless `fast_dev_run=True`."""
    tutils.reset_seed()

    class CurrentModel(LightTrainDataloader, LightValidationMixin, TestModelBase):

        validation_step_invoked = False
        validation_epoch_end_invoked = False

        def validation_step(self, *args, **kwargs):
            self.validation_step_invoked = True
            return super().validation_step(*args, **kwargs)

        def validation_epoch_end(self, *args, **kwargs):
            self.validation_epoch_end_invoked = True
            return super().validation_epoch_end(*args, **kwargs)

    hparams = tutils.get_default_hparams()
    model = CurrentModel(hparams)

    trainer_options = dict(
        show_progress_bar=False,
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
    model = CurrentModel(hparams)
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
    test_step = 8

    class InfLossModel(LightTrainDataloader, TestModelBase):

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            if batch_idx == test_step:
                if isinstance(output, dict):
                    output['loss'] *= torch.tensor(math.inf)  # make loss infinite
                else:
                    output /= 0
            return output

    hparams = tutils.get_default_hparams()
    model = InfLossModel(hparams)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=(test_step + 1),
        terminate_on_nan=True
    )

    with pytest.raises(ValueError, match=r'.*The loss returned in `training_step` is nan or inf.*'):
        trainer.fit(model)
        assert trainer.global_step == test_step

    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_nan_params_detection(tmpdir):
    test_step = 8

    class NanParamModel(LightTrainDataloader, TestModelBase):

        def on_after_backward(self):
            if self.global_step == test_step:
                # simulate parameter that became nan
                torch.nn.init.constant_(self.c_d1.bias, math.nan)

    hparams = tutils.get_default_hparams()

    model = NanParamModel(hparams)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=(test_step + 1),
        terminate_on_nan=True
    )

    with pytest.raises(ValueError, match=r'.*Detected nan and/or inf values in `c_d1.bias`.*'):
        trainer.fit(model)
        assert trainer.global_step == test_step

    # after aborting the training loop, model still has nan-valued params
    params = torch.cat([param.view(-1) for param in model.parameters()])
    assert not torch.isfinite(params).all()


def test_trainer_interrupted_flag(tmpdir):
    """Test the flag denoting that a user interrupted training."""

    model = DictHparamsModel({'in_features': 28 * 28, 'out_features': 10})

    class InterruptCallback(Callback):
        def __init__(self):
            super().__init__()

        def on_batch_start(self, trainer, pl_module):
            raise KeyboardInterrupt

    interrupt_callback = InterruptCallback()

    trainer_options = {
        'callbacks': [interrupt_callback],
        'max_epochs': 1,
        'val_percent_check': 0.1,
        'train_percent_check': 0.2,
        'progress_bar_refresh_rate': 0,
        'logger': False,
        'default_root_dir': tmpdir,
    }

    trainer = Trainer(**trainer_options)
    assert not trainer.interrupted
    trainer.fit(model)
    assert trainer.interrupted


def test_gradient_clipping(tmpdir):
    """
    Test gradient clipping
    """
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    # test that gradient is clipped correctly
    def _optimizer_step(*args, **kwargs):
        parameters = model.parameters()
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
        assert (grad_norm - 1.0).abs() < 0.01, "Gradient norm != 1.0: {grad_norm}".format(grad_norm=grad_norm)

    trainer = Trainer(max_steps=1,
                      max_epochs=1,
                      gradient_clip_val=1.0,
                      default_root_dir=tmpdir)

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
