from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule
from tests.base import EvalModelTemplate


def test_trainer_callback_system(tmpdir):
    """Test the callback system."""

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    def _check_args(trainer, pl_module):
        assert isinstance(trainer, Trainer)
        assert isinstance(pl_module, LightningModule)

    class TestCallback(Callback):
        def __init__(self):
            super().__init__()
            self.setup_called = False
            self.teardown_called = False
            self.on_init_start_called = False
            self.on_init_end_called = False
            self.on_fit_start_called = False
            self.on_fit_end_called = False
            self.on_sanity_check_start_called = False
            self.on_sanity_check_end_called = False
            self.on_epoch_start_called = False
            self.on_epoch_end_called = False
            self.on_batch_start_called = False
            self.on_batch_end_called = False
            self.on_train_batch_start_called = False
            self.on_train_batch_end_called = False
            self.on_validation_batch_start_called = False
            self.on_validation_batch_end_called = False
            self.on_test_batch_start_called = False
            self.on_test_batch_end_called = False
            self.on_train_start_called = False
            self.on_train_end_called = False
            self.on_pretrain_routine_start_called = False
            self.on_pretrain_routine_end_called = False
            self.on_validation_start_called = False
            self.on_validation_end_called = False
            self.on_test_start_called = False
            self.on_test_end_called = False

        def setup(self, trainer, pl_module, stage: str):
            assert isinstance(trainer, Trainer)
            self.setup_called = True

        def teardown(self, trainer, pl_module, step: str):
            assert isinstance(trainer, Trainer)
            self.teardown_called = True

        def on_init_start(self, trainer):
            assert isinstance(trainer, Trainer)
            self.on_init_start_called = True

        def on_init_end(self, trainer):
            assert isinstance(trainer, Trainer)
            self.on_init_end_called = True

        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer, Trainer)
            self.on_fit_start_called = True

        def on_fit_end(self, trainer, pl_module):
            assert isinstance(trainer, Trainer)
            self.on_fit_end_called = True

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

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            _check_args(trainer, pl_module)
            self.on_train_batch_start_called = True

        def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            _check_args(trainer, pl_module)
            self.on_train_batch_end_called = True

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            _check_args(trainer, pl_module)
            self.on_validation_batch_start_called = True

        def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            _check_args(trainer, pl_module)
            self.on_validation_batch_end_called = True

        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            _check_args(trainer, pl_module)
            self.on_test_batch_start_called = True

        def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            _check_args(trainer, pl_module)
            self.on_test_batch_end_called = True

        def on_train_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_train_start_called = True

        def on_train_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_train_end_called = True

        def on_pretrain_routine_start(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_pretrain_routine_start_called = True

        def on_pretrain_routine_end(self, trainer, pl_module):
            _check_args(trainer, pl_module)
            self.on_pretrain_routine_end_called = True

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

    trainer_options = dict(
        default_root_dir=tmpdir,
        callbacks=[test_callback],
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        progress_bar_refresh_rate=0,
    )

    assert not test_callback.setup_called
    assert not test_callback.teardown_called
    assert not test_callback.on_init_start_called
    assert not test_callback.on_init_end_called
    assert not test_callback.on_fit_start_called
    assert not test_callback.on_fit_end_called
    assert not test_callback.on_sanity_check_start_called
    assert not test_callback.on_sanity_check_end_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_batch_start_called
    assert not test_callback.on_batch_end_called
    assert not test_callback.on_train_batch_start_called
    assert not test_callback.on_train_batch_end_called
    assert not test_callback.on_validation_batch_start_called
    assert not test_callback.on_validation_batch_end_called
    assert not test_callback.on_test_batch_start_called
    assert not test_callback.on_test_batch_end_called
    assert not test_callback.on_train_start_called
    assert not test_callback.on_train_end_called
    assert not test_callback.on_pretrain_routine_start_called
    assert not test_callback.on_pretrain_routine_end_called
    assert not test_callback.on_validation_start_called
    assert not test_callback.on_validation_end_called
    assert not test_callback.on_test_start_called
    assert not test_callback.on_test_end_called

    # fit model
    trainer = Trainer(**trainer_options)

    assert trainer.callbacks[0] == test_callback
    assert test_callback.on_init_start_called
    assert test_callback.on_init_end_called
    assert not test_callback.setup_called
    assert not test_callback.teardown_called
    assert not test_callback.on_fit_start_called
    assert not test_callback.on_fit_end_called
    assert not test_callback.on_sanity_check_start_called
    assert not test_callback.on_sanity_check_end_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_epoch_start_called
    assert not test_callback.on_batch_start_called
    assert not test_callback.on_batch_end_called
    assert not test_callback.on_train_batch_start_called
    assert not test_callback.on_train_batch_end_called
    assert not test_callback.on_validation_batch_start_called
    assert not test_callback.on_validation_batch_end_called
    assert not test_callback.on_test_batch_start_called
    assert not test_callback.on_test_batch_end_called
    assert not test_callback.on_train_start_called
    assert not test_callback.on_train_end_called
    assert not test_callback.on_pretrain_routine_start_called
    assert not test_callback.on_pretrain_routine_end_called
    assert not test_callback.on_validation_start_called
    assert not test_callback.on_validation_end_called
    assert not test_callback.on_test_start_called
    assert not test_callback.on_test_end_called

    trainer.fit(model)

    assert test_callback.setup_called
    assert test_callback.teardown_called
    assert test_callback.on_init_start_called
    assert test_callback.on_init_end_called
    assert test_callback.on_fit_start_called
    assert test_callback.on_fit_end_called
    assert test_callback.on_sanity_check_start_called
    assert test_callback.on_sanity_check_end_called
    assert test_callback.on_epoch_start_called
    assert test_callback.on_epoch_start_called
    assert test_callback.on_batch_start_called
    assert test_callback.on_batch_end_called
    assert test_callback.on_train_batch_start_called
    assert test_callback.on_train_batch_end_called
    assert test_callback.on_validation_batch_start_called
    assert test_callback.on_validation_batch_end_called
    assert test_callback.on_train_start_called
    assert test_callback.on_train_end_called
    assert test_callback.on_pretrain_routine_start_called
    assert test_callback.on_pretrain_routine_end_called
    assert test_callback.on_validation_start_called
    assert test_callback.on_validation_end_called
    assert not test_callback.on_test_batch_start_called
    assert not test_callback.on_test_batch_end_called
    assert not test_callback.on_test_start_called
    assert not test_callback.on_test_end_called

    # reset setup teardown callback
    test_callback.teardown_called = False
    test_callback.setup_called = False

    test_callback = TestCallback()
    trainer_options.update(callbacks=[test_callback])
    trainer = Trainer(**trainer_options)
    trainer.test(model)

    assert test_callback.setup_called
    assert test_callback.teardown_called
    assert test_callback.on_test_batch_start_called
    assert test_callback.on_test_batch_end_called
    assert test_callback.on_test_start_called
    assert test_callback.on_test_end_called
    assert not test_callback.on_validation_start_called
    assert not test_callback.on_validation_end_called
    assert not test_callback.on_validation_batch_end_called
    assert not test_callback.on_validation_batch_start_called
