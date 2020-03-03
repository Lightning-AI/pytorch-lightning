import math
import os
import pytest
import torch
import argparse

import tests.models.utils as tutils
from unittest import mock
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tests.models import (
    TestModelBase,
    LightningTestModel,
    LightEmptyTestStep,
    LightValidationStepMixin,
    LightValidationMultipleDataloadersMixin,
    LightTrainDataloader,
    LightTestDataloader,
    LightValidationMixin,
    LightTestMixin
)
from pytorch_lightning.core.lightning import load_hparams_from_tags_csv
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.utilities.debugging import MisconfigurationException
from pytorch_lightning import Callback


def test_no_val_module(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()

    class CurrentTestModel(LightTrainDataloader, TestModelBase):
        pass

    model = CurrentTestModel(hparams)

    # logger file to get meta
    logger = tutils.get_test_tube_logger(tmpdir, False)

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

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    logger = tutils.get_test_tube_logger(tmpdir, False)

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
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
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

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)
    schedule = {1: 2, 3: 4}

    trainer = Trainer(accumulate_grad_batches=schedule,
                      train_percent_check=0.1,
                      val_percent_check=0.1,
                      max_epochs=4,
                      default_save_path=tmpdir)

    # for the test
    trainer.optimizer_step = optimizer_step
    model.prev_called_batch_idx = 0

    trainer.fit(model)


def test_loading_meta_tags(tmpdir):
    tutils.reset_seed()

    from argparse import Namespace
    hparams = tutils.get_hparams()

    # save tags
    logger = tutils.get_test_tube_logger(tmpdir, False)
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


def test_model_checkpoint_options(tmp_path):
    """Test ModelCheckpoint options."""
    def mock_save_function(filepath):
        open(filepath, 'a').close()

    hparams = tutils.get_hparams()
    _ = LightningTestModel(hparams)

    # simulated losses
    save_dir = tmp_path / "1"
    save_dir.mkdir()
    losses = [10, 9, 2.8, 5, 2.5]

    # -----------------
    # CASE K=-1  (all)
    checkpoint_callback = ModelCheckpoint(save_dir, save_top_k=-1, verbose=1)
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for i, loss in enumerate(losses):
        trainer.current_epoch = i
        trainer.callback_metrics = {'val_loss': loss}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == len(losses), "Should save all models when save_top_k=-1"

    # verify correct naming
    for i in range(0, len(losses)):
        assert f"_ckpt_epoch_{i}.ckpt" in file_lists

    save_dir = tmp_path / "2"
    save_dir.mkdir()

    # -----------------
    # CASE K=0 (none)
    checkpoint_callback = ModelCheckpoint(save_dir, save_top_k=0, verbose=1)
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for i, loss in enumerate(losses):
        trainer.current_epoch = i
        trainer.callback_metrics = {'val_loss': loss}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

    file_lists = os.listdir(save_dir)

    assert len(file_lists) == 0, "Should save 0 models when save_top_k=0"

    save_dir = tmp_path / "3"
    save_dir.mkdir()

    # -----------------
    # CASE K=1 (2.5, epoch 4)
    checkpoint_callback = ModelCheckpoint(save_dir, save_top_k=1, verbose=1, prefix='test_prefix')
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for i, loss in enumerate(losses):
        trainer.current_epoch = i
        trainer.callback_metrics = {'val_loss': loss}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == 1, "Should save 1 model when save_top_k=1"
    assert 'test_prefix_ckpt_epoch_4.ckpt' in file_lists

    save_dir = tmp_path / "4"
    save_dir.mkdir()

    # -----------------
    # CASE K=2 (2.5 epoch 4, 2.8 epoch 2)
    # make sure other files don't get deleted

    checkpoint_callback = ModelCheckpoint(save_dir, save_top_k=2, verbose=1)
    open(f"{save_dir}/other_file.ckpt", 'a').close()
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for i, loss in enumerate(losses):
        trainer.current_epoch = i
        trainer.callback_metrics = {'val_loss': loss}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == 3, 'Should save 2 model when save_top_k=2'
    assert '_ckpt_epoch_4.ckpt' in file_lists
    assert '_ckpt_epoch_2.ckpt' in file_lists
    assert 'other_file.ckpt' in file_lists

    save_dir = tmp_path / "5"
    save_dir.mkdir()

    # -----------------
    # CASE K=4 (save all 4 models)
    # multiple checkpoints within same epoch

    checkpoint_callback = ModelCheckpoint(save_dir, save_top_k=4, verbose=1)
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for loss in losses:
        trainer.current_epoch = 0
        trainer.callback_metrics = {'val_loss': loss}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == 4, 'Should save all 4 models when save_top_k=4 within same epoch'

    save_dir = tmp_path / "6"
    save_dir.mkdir()

    # -----------------
    # CASE K=3 (save the 2nd, 3rd, 4th model)
    # multiple checkpoints within same epoch

    checkpoint_callback = ModelCheckpoint(save_dir, save_top_k=3, verbose=1)
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for loss in losses:
        trainer.current_epoch = 0
        trainer.callback_metrics = {'val_loss': loss}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == 3, 'Should save 3 models when save_top_k=3'
    assert '_ckpt_epoch_0_v2.ckpt' in file_lists
    assert '_ckpt_epoch_0_v1.ckpt' in file_lists
    assert '_ckpt_epoch_0.ckpt' in file_lists


def test_model_freeze_unfreeze():
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    model.freeze()
    model.unfreeze()


def test_resume_from_checkpoint_epoch_restored(tmpdir):
    """Verify resuming from checkpoint runs the right number of epochs"""
    import types

    tutils.reset_seed()

    hparams = tutils.get_hparams()

    def new_model():
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

    model = new_model()

    trainer_options = dict(
        show_progress_bar=False,
        max_epochs=2,
        train_percent_check=0.65,
        val_percent_check=1,
        checkpoint_callback=ModelCheckpoint(tmpdir, save_top_k=-1),
        logger=False,
        default_save_path=tmpdir,
        early_stop_callback=False,
        val_check_interval=0.5,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    training_batches = trainer.num_training_batches

    assert model.num_epochs_seen == 2
    assert model.num_batches_seen == training_batches * 2

    # Other checkpoints can be uncommented if/when resuming mid-epoch is supported
    checkpoints = [
        # os.path.join(trainer.checkpoint_callback.filepath, "_ckpt_epoch_0.ckpt"),
        os.path.join(trainer.checkpoint_callback.filepath, "_ckpt_epoch_0_v0.ckpt"),
        # os.path.join(trainer.checkpoint_callback.filepath, "_ckpt_epoch_1.ckpt"),
        os.path.join(trainer.checkpoint_callback.filepath, "_ckpt_epoch_1_v0.ckpt"),
    ]

    for check in checkpoints:
        next_model = new_model()
        state = torch.load(check)

        # Resume training
        trainer_options['max_epochs'] = 4
        new_trainer = Trainer(**trainer_options, resume_from_checkpoint=check)
        new_trainer.fit(next_model)
        assert state['global_step'] + next_model.num_batches_seen == training_batches * 4


def _init_steps_model():
    """private method for initializing a model with 5% train epochs"""
    tutils.reset_seed()
    model, _ = tutils.get_model()

    # define train epoch to 5% of data
    train_percent = 0.05
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
        default_save_path=tmpdir,
        max_epochs=5,
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
    assert trainer.global_step == num_train_samples * trainer.max_epochs \
        and trainer.current_epoch == trainer.max_epochs - 1, "Model did not stop at max_epochs"


def test_trainer_min_steps_and_epochs(tmpdir):
    """Verify model trains according to specified min steps"""
    model, trainer_options, num_train_samples = _init_steps_model()

    # define callback for stopping the model and default epochs
    trainer_options.update(dict(
        default_save_path=tmpdir,
        early_stop_callback=EarlyStopping(monitor='val_loss', min_delta=1.0),
        val_check_interval=20,
        min_epochs=1,
        max_epochs=10
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

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # verify torch.backends.cudnn.benchmark is not turned on
    assert not torch.backends.cudnn.benchmark

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
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
    hparams = tutils.get_hparams()

    class LocalModel(LightTrainDataloader, TestModelBase):
        pass

    class LocalModelNoEnd(LightTrainDataloader, LightTestDataloader, LightEmptyTestStep, TestModelBase):
        pass

    class LocalModelNoStep(LightTrainDataloader, TestModelBase):
        def test_end(self, outputs):
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

@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(**Trainer.default_attributes))
def test_default_args(tmpdir):
    """Tests default argument parser for Trainer"""
    tutils.reset_seed()

    # logger file to get meta
    logger = tutils.get_test_tube_logger(tmpdir, False)

    parser = argparse.ArgumentParser(add_help=False)
    args = parser.parse_args()
    args.logger = logger

    args.max_epochs = 5
    trainer = Trainer.from_argparse_args(args)

    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 5
