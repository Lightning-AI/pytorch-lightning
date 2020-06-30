from pathlib import Path

import pytest
import torch

import tests.base.develop_utils as tutils
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tests.base import EvalModelTemplate


def test_early_stopping_functionality(tmpdir):

    class CurrentModel(EvalModelTemplate):
        def validation_epoch_end(self, outputs):
            losses = [8, 4, 2, 3, 4, 5, 8, 10]
            val_loss = losses[self.current_epoch]
            return {'val_loss': torch.tensor(val_loss)}

    model = CurrentModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=True,
        overfit_batches=0.20,
        max_epochs=20,
    )
    result = trainer.fit(model)
    print(trainer.current_epoch)

    assert trainer.current_epoch == 5, 'early_stopping failed'


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

        def setup(self, trainer, stage: str):
            assert isinstance(trainer, Trainer)
            self.setup_called = True

        def teardown(self, trainer, step: str):
            assert isinstance(trainer, Trainer)
            self.teardown_called = True

        def on_init_start(self, trainer):
            assert isinstance(trainer, Trainer)
            self.on_init_start_called = True

        def on_init_end(self, trainer):
            assert isinstance(trainer, Trainer)
            self.on_init_end_called = True

        def on_fit_start(self, trainer):
            assert isinstance(trainer, Trainer)
            self.on_fit_start_called = True

        def on_fit_end(self, trainer):
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


def test_early_stopping_no_val_step(tmpdir):
    """Test that early stopping callback falls back to training metrics when no validation defined."""

    class CurrentModel(EvalModelTemplate):
        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            output.update({'my_train_metric': output['loss']})  # could be anything else
            return output

    model = CurrentModel()
    model.validation_step = None
    model.val_dataloader = None

    stopping = EarlyStopping(monitor='my_train_metric', min_delta=0.1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=stopping,
        overfit_batches=0.20,
        max_epochs=2,
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'
    assert trainer.current_epoch < trainer.max_epochs


def test_pickling(tmpdir):
    import pickle
    early_stopping = EarlyStopping()
    ckpt = ModelCheckpoint(tmpdir)

    early_stopping_pickled = pickle.dumps(early_stopping)
    ckpt_pickled = pickle.dumps(ckpt)

    early_stopping_loaded = pickle.loads(early_stopping_pickled)
    ckpt_loaded = pickle.loads(ckpt_pickled)

    assert vars(early_stopping) == vars(early_stopping_loaded)
    assert vars(ckpt) == vars(ckpt_loaded)


@pytest.mark.parametrize('save_top_k', [-1, 0, 1, 2])
def test_model_checkpoint_with_non_string_input(tmpdir, save_top_k):
    """ Test that None in checkpoint callback is valid and that chkp_path is set correctly """
    tutils.reset_seed()
    model = EvalModelTemplate()

    checkpoint = ModelCheckpoint(filepath=None, save_top_k=save_top_k)

    trainer = Trainer(
        default_root_dir=tmpdir,
        checkpoint_callback=checkpoint,
        overfit_batches=0.20,
        max_epochs=2,
    )
    trainer.fit(model)

    # These should be different if the dirpath has be overridden
    assert trainer.ckpt_path != trainer.default_root_dir


@pytest.mark.parametrize(
    'logger_version,expected',
    [(None, 'version_0'), (1, 'version_1'), ('awesome', 'awesome')],
)
def test_model_checkpoint_path(tmpdir, logger_version, expected):
    """Test that "version_" prefix is only added when logger's version is an integer"""
    tutils.reset_seed()
    model = EvalModelTemplate()
    logger = TensorBoardLogger(str(tmpdir), version=logger_version)

    trainer = Trainer(
        default_root_dir=tmpdir,
        overfit_batches=0.2,
        max_epochs=2,
        logger=logger,
    )
    trainer.fit(model)

    ckpt_version = Path(trainer.ckpt_path).parent.name
    assert ckpt_version == expected
