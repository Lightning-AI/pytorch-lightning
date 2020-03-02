import tests.models.utils as tutils
from pytorch_lightning import Trainer, LightningModule
from tests.models import (
    TestModelBase,
    LightTrainDataloader,
    LightValidationMixin,
    LightTestMixin
)

from pytorch_lightning import Callback


def test_trainer_callback_system(tmpdir):
    """Test the callback system."""

    class CurrentTestModel(
        LightTrainDataloader,
        LightTestMixin,
        LightValidationMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_hparams()
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
        'show_progress_bar': False
    }

    assert not test_callback.on_init_start_called
    assert not test_callback.on_init_end_called

    # fit model
    trainer = Trainer(**trainer_options)

    assert trainer.callbacks[0] == test_callback
    assert test_callback.on_init_start_called
    assert test_callback.on_init_end_called
    assert not test_callback.on_train_start_called

    trainer.fit(model)

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
