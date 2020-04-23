import pytest

import tests.base.utils as tutils
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ProgressBarBase, ProgressBar, ModelCheckpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import (
    LightTrainDataloader,
    LightTestMixin,
    LightValidationMixin,
    TestModelBase
)


def test_trainer_callback_system(tmpdir):
    """Test the callback system."""

    class CurrentTestModel(
        LightTrainDataloader,
        LightTestMixin,
        LightValidationMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    def _check_args(trainer, pl_module):
        assert isinstance(trainer, Trainer)
        assert isinstance(pl_module, LightningModule)

    class TestCallback(Callback):
        def __init__(self):
            super().__init__()
            self.on_init_start_called = False
            self.on_init_end_called = False
            self.on_sanity_check_start_called = False
            self.on_sanity_check_end_called = False
            self.on_epoch_start_called = False
            self.on_epoch_end_called = False
            self.on_batch_start_called = False
            self.on_batch_end_called = False
            self.on_validation_batch_start_called = False
            self.on_validation_batch_end_called = False
            self.on_test_batch_start_called = False
            self.on_test_batch_end_called = False
            self.on_train_start_called = False
            self.on_train_end_called = False
            self.on_validation_start_called = False
            self.on_validation_end_called = False
            self.on_test_start_called = False
            self.on_test_end_called = False

        def on_init_start(self, trainer):
            assert isinstance(trainer, Trainer)
            self.on_init_start_called = True

        def on_init_end(self, trainer):
            assert isinstance(trainer, Trainer)
            self.on_init_end_called = True

        def on_sanity_check_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_sanity_check_start_called = True

        def on_sanity_check_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_sanity_check_end_called = True

        def on_epoch_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_epoch_start_called = True

        def on_epoch_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_epoch_end_called = True

        def on_batch_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_batch_start_called = True

        def on_batch_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_batch_end_called = True

        def on_validation_batch_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_validation_batch_start_called = True

        def on_validation_batch_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_validation_batch_end_called = True

        def on_test_batch_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_test_batch_start_called = True

        def on_test_batch_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_test_batch_end_called = True

        def on_train_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_train_start_called = True

        def on_train_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_train_end_called = True

        def on_validation_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_validation_start_called = True

        def on_validation_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_validation_end_called = True

        def on_test_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_test_start_called = True

        def on_test_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_test_end_called = True

    test_callback = TestCallback()

    trainer_options = {
        'callbacks': [test_callback],
        'max_epochs': 1,
        'val_percent_check': 0.1,
        'train_percent_check': 0.2,
        'progress_bar_refresh_rate': 0
    }

    assert not test_callback.on_init_start_called
    assert not test_callback.on_init_end_called
    assert not test_callback.on_sanity_check_start_called
    assert not test_callback.on_sanity_check_end_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_batch_start_called
    assert not test_callback.on_batch_end_called
    assert not test_callback.on_validation_batch_start_called
    assert not test_callback.on_validation_batch_end_called
    assert not test_callback.on_test_batch_start_called
    assert not test_callback.on_test_batch_end_called
    assert not test_callback.on_train_start_called
    assert not test_callback.on_train_end_called
    assert not test_callback.on_validation_start_called
    assert not test_callback.on_validation_end_called
    assert not test_callback.on_test_start_called
    assert not test_callback.on_test_end_called

    # fit model
    trainer = Trainer(**trainer_options)

    assert trainer.callbacks[0] == test_callback
    assert test_callback.on_init_start_called
    assert test_callback.on_init_end_called
    assert not test_callback.on_sanity_check_start_called
    assert not test_callback.on_sanity_check_end_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_batch_start_called
    assert not test_callback.on_batch_end_called
    assert not test_callback.on_validation_batch_start_called
    assert not test_callback.on_validation_batch_end_called
    assert not test_callback.on_test_batch_start_called
    assert not test_callback.on_test_batch_end_called
    assert not test_callback.on_train_start_called
    assert not test_callback.on_train_end_called
    assert not test_callback.on_validation_start_called
    assert not test_callback.on_validation_end_called
    assert not test_callback.on_test_start_called
    assert not test_callback.on_test_end_called

    trainer.fit(model)

    assert test_callback.on_init_start_called
    assert test_callback.on_init_end_called
    assert test_callback.on_sanity_check_start_called
    assert test_callback.on_sanity_check_end_called
    assert test_callback.on_epoch_start_called
    assert test_callback.on_epoch_start_called
    assert test_callback.on_batch_start_called
    assert test_callback.on_batch_end_called
    assert test_callback.on_validation_batch_start_called
    assert test_callback.on_validation_batch_end_called
    assert test_callback.on_train_start_called
    assert test_callback.on_train_end_called
    assert test_callback.on_validation_start_called
    assert test_callback.on_validation_end_called
    assert not test_callback.on_test_batch_start_called
    assert not test_callback.on_test_batch_end_called
    assert not test_callback.on_test_start_called
    assert not test_callback.on_test_end_called

    test_callback = TestCallback()
    trainer_options['callbacks'] = [test_callback]
    trainer = Trainer(**trainer_options)
    trainer.test(model)

    assert test_callback.on_test_batch_start_called
    assert test_callback.on_test_batch_end_called
    assert test_callback.on_test_start_called
    assert test_callback.on_test_end_called
    assert not test_callback.on_validation_start_called
    assert not test_callback.on_validation_end_called
    assert not test_callback.on_validation_batch_end_called
    assert not test_callback.on_validation_batch_start_called


def test_early_stopping_no_val_step(tmpdir):
    """Test that early stopping callback falls back to training metrics when no validation defined."""
    tutils.reset_seed()

    class ModelWithoutValStep(LightTrainDataloader, TestModelBase):

        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            loss = output['loss']  # could be anything else
            output.update({'my_train_metric': loss})
            return output

    hparams = tutils.get_default_hparams()
    model = ModelWithoutValStep(hparams)

    stopping = EarlyStopping(monitor='my_train_metric', min_delta=0.1)
    trainer_options = dict(
        default_root_dir=tmpdir,
        early_stop_callback=stopping,
        overfit_pct=0.20,
        max_epochs=5,
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'
    assert trainer.current_epoch < trainer.max_epochs


def test_model_checkpoint_with_non_string_input(tmpdir):
    """ Test that None in checkpoint callback is valid and that chkp_path is
        set correctly """
    tutils.reset_seed()

    class CurrentTestModel(LightTrainDataloader, TestModelBase):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    checkpoint = ModelCheckpoint(filepath=None, save_top_k=-1)

    trainer = Trainer(default_root_dir=tmpdir,
                      checkpoint_callback=checkpoint,
                      overfit_pct=0.20,
                      max_epochs=5
                      )
    result = trainer.fit(model)

    # These should be different if the dirpath has be overridden
    assert trainer.ckpt_path != trainer.default_root_dir


@pytest.mark.parametrize('callbacks,refresh_rate', [
    ([], 1),
    ([], 2),
    ([ProgressBar(refresh_rate=1)], 0),
    ([ProgressBar(refresh_rate=2)], 0),
    ([ProgressBar(refresh_rate=2)], 1),
])
def test_progress_bar_on(callbacks, refresh_rate):
    """Test different ways the progress bar can be turned on."""

    trainer = Trainer(
        callbacks=callbacks,
        progress_bar_refresh_rate=refresh_rate,
        max_epochs=1,
        overfit_pct=0.2,
    )

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBarBase)]
    # Trainer supports only a single progress bar callback at the moment
    assert len(progress_bars) == 1
    assert progress_bars[0] is trainer.progress_bar_callback


@pytest.mark.parametrize('callbacks,refresh_rate', [
    ([], 0),
    ([], False),
    ([ModelCheckpoint('.')], 0),
])
def test_progress_bar_off(callbacks, refresh_rate):
    """Test different ways the progress bar can be turned off."""

    trainer = Trainer(
        callbacks=callbacks,
        progress_bar_refresh_rate=refresh_rate,
    )

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBar)]
    assert 0 == len(progress_bars)
    assert not trainer.progress_bar_callback


def test_progress_bar_misconfiguration():
    """Test that Trainer doesn't accept multiple progress bars."""
    callbacks = [ProgressBar(), ProgressBar(), ModelCheckpoint('.')]
    with pytest.raises(MisconfigurationException, match=r'^You added multiple progress bar callbacks'):
        Trainer(callbacks=callbacks)


def test_progress_bar_totals():
    """Test that the progress finishes with the correct total steps processed."""

    class CurrentTestModel(
        LightTrainDataloader,
        LightTestMixin,
        LightValidationMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    trainer = Trainer(
        progress_bar_refresh_rate=1,
        val_percent_check=1.0,
        max_epochs=1,
    )
    bar = trainer.progress_bar_callback
    assert 0 == bar.total_train_batches
    assert 0 == bar.total_val_batches
    assert 0 == bar.total_test_batches

    trainer.fit(model)

    # check main progress bar total
    n = bar.total_train_batches
    m = bar.total_val_batches
    assert len(trainer.train_dataloader) == n
    assert bar.main_progress_bar.total == n + m

    # check val progress bar total
    assert sum(len(loader) for loader in trainer.val_dataloaders) == m
    assert bar.val_progress_bar.total == m

    # main progress bar should have reached the end (train batches + val batches)
    assert bar.main_progress_bar.n == n + m
    assert bar.train_batch_idx == n

    # val progress bar should have reached the end
    assert bar.val_progress_bar.n == m
    assert bar.val_batch_idx == m

    # check that the test progress bar is off
    assert 0 == bar.total_test_batches
    assert bar.test_progress_bar is None

    trainer.test(model)

    # check test progress bar total
    k = bar.total_test_batches
    assert sum(len(loader) for loader in trainer.test_dataloaders) == k
    assert bar.test_progress_bar.total == k

    # test progress bar should have reached the end
    assert bar.test_progress_bar.n == k
    assert bar.test_batch_idx == k


def test_progress_bar_fast_dev_run():
    class CurrentTestModel(
        LightTrainDataloader,
        LightTestMixin,
        LightValidationMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    trainer = Trainer(
        fast_dev_run=True,
    )

    progress_bar = trainer.progress_bar_callback
    assert 1 == progress_bar.total_train_batches
    # total val batches are known only after val dataloaders have reloaded

    trainer.fit(model)

    assert 1 == progress_bar.total_val_batches
    assert 1 == progress_bar.train_batch_idx
    assert 1 == progress_bar.val_batch_idx
    assert 0 == progress_bar.test_batch_idx

    # the main progress bar should display 2 batches (1 train, 1 val)
    assert 2 == progress_bar.main_progress_bar.total
    assert 2 == progress_bar.main_progress_bar.n


@pytest.mark.parametrize('refresh_rate', [0, 1, 50])
def test_progress_bar_progress_refresh(refresh_rate):
    """Test that the three progress bars get correctly updated when using different refresh rates."""

    class CurrentTestModel(
        LightTrainDataloader,
        LightTestMixin,
        LightValidationMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    class CurrentProgressBar(ProgressBar):

        train_batches_seen = 0
        val_batches_seen = 0
        test_batches_seen = 0

        def on_batch_start(self, trainer, pl_module):
            super().on_batch_start(trainer, pl_module)
            assert self.train_batch_idx == trainer.batch_idx

        def on_batch_end(self, trainer, pl_module):
            super().on_batch_end(trainer, pl_module)
            assert self.train_batch_idx == trainer.batch_idx + 1
            if not self.disabled and self.train_batch_idx % self.refresh_rate == 0:
                assert self.main_progress_bar.n == self.train_batch_idx
            self.train_batches_seen += 1

        def on_validation_batch_end(self, trainer, pl_module):
            super().on_validation_batch_end(trainer, pl_module)
            if not self.disabled and self.val_batch_idx % self.refresh_rate == 0:
                assert self.val_progress_bar.n == self.val_batch_idx
            self.val_batches_seen += 1

        def on_test_batch_end(self, trainer, pl_module):
            super().on_test_batch_end(trainer, pl_module)
            if not self.disabled and self.test_batch_idx % self.refresh_rate == 0:
                assert self.test_progress_bar.n == self.test_batch_idx
            self.test_batches_seen += 1

    progress_bar = CurrentProgressBar(refresh_rate=refresh_rate)
    trainer = Trainer(
        callbacks=[progress_bar],
        progress_bar_refresh_rate=101,  # should not matter if custom callback provided
        train_percent_check=1.0,
        num_sanity_val_steps=2,
        max_epochs=3,
    )
    assert trainer.progress_bar_callback.refresh_rate == refresh_rate != trainer.progress_bar_refresh_rate

    trainer.fit(model)
    assert progress_bar.train_batches_seen == 3 * progress_bar.total_train_batches
    assert progress_bar.val_batches_seen == 3 * progress_bar.total_val_batches + trainer.num_sanity_val_steps

    trainer.test(model)
    assert progress_bar.test_batches_seen == progress_bar.total_test_batches


def test_model_checkpoint_with_non_string_input(tmpdir):
    """ Test that None in checkpoint callback is valid and that chkp_path is
        set correctly """
    tutils.reset_seed()

    class CurrentTestModel(LightTrainDataloader, TestModelBase):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    checkpoint = ModelCheckpoint(filepath=None, save_top_k=-1)

    trainer = Trainer(default_root_dir=tmpdir,
                      checkpoint_callback=checkpoint,
                      overfit_pct=0.20,
                      max_epochs=5
                      )
    result = trainer.fit(model)

    # These should be different if the dirpath has be overridden
    assert trainer.ckpt_path != trainer.default_root_dir
