import tests.base.utils as tutils
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
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
            self.on_epoch_start_called = False
            self.on_epoch_end_called = False
            self.on_batch_start_called = False
            self.on_batch_end_called = False
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
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_batch_start_called
    assert not test_callback.on_batch_end_called
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
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_batch_start_called
    assert not test_callback.on_batch_end_called
    assert not test_callback.on_train_start_called
    assert not test_callback.on_train_end_called
    assert not test_callback.on_validation_start_called
    assert not test_callback.on_validation_end_called
    assert not test_callback.on_test_start_called
    assert not test_callback.on_test_end_called

    trainer.fit(model)

    assert test_callback.on_init_start_called
    assert test_callback.on_init_end_called
    assert test_callback.on_epoch_start_called
    assert test_callback.on_epoch_start_called
    assert test_callback.on_batch_start_called
    assert test_callback.on_batch_end_called
    assert test_callback.on_train_start_called
    assert test_callback.on_train_end_called
    assert test_callback.on_validation_start_called
    assert test_callback.on_validation_end_called
    assert not test_callback.on_test_start_called
    assert not test_callback.on_test_end_called

    trainer.test()

    assert test_callback.on_test_start_called
    assert test_callback.on_test_end_called


def test_early_stopping_without_val_step(tmpdir):
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
