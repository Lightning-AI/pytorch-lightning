import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.base import DummyLogger
from tests.base import BoringModel


@pytest.mark.parametrize('tuner_alg', ['batch size scaler', 'learning rate finder'])
def test_skip_on_fast_dev_run_tuner(tmpdir, tuner_alg):
    """ Test that tuner algorithms are skipped if fast dev run is enabled """

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_scale_batch_size=True if tuner_alg == 'batch size scaler' else False,
        auto_lr_find=True if tuner_alg == 'learning rate finder' else False,
        fast_dev_run=True
    )
    expected_message = f'Skipping {tuner_alg} since fast_dev_run is enabled.'
    with pytest.warns(UserWarning, match=expected_message):
        trainer.tune(model)


@pytest.mark.parametrize('fast_dev_run', [1, 4])
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_callbacks_and_logger_not_called_with_fastdevrun(tmpdir, fast_dev_run):
    """
    Test that ModelCheckpoint, EarlyStopping and Logger are turned off with fast_dev_run
    """
    class FastDevRunModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.training_step_called = False
            self.validation_step_called = False
            self.test_step_called = False

        def training_step(self, batch, batch_idx):
            self.log('some_metric', torch.tensor(7.))
            self.logger.experiment.dummy_log('some_distribution', torch.randn(7) + batch_idx)
            self.training_step_called = True
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True
            return super().validation_step(batch, batch_idx)

    checkpoint_callback = ModelCheckpoint()
    early_stopping_callback = EarlyStopping()
    trainer_config = dict(
        fast_dev_run=fast_dev_run,
        logger=True,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    def _make_fast_dev_run_assertions(trainer):
        # there should be no logger with fast_dev_run
        assert isinstance(trainer.logger, DummyLogger)
        assert len(trainer.dev_debugger.logged_metrics) == fast_dev_run

        # checkpoint callback should not have been called with fast_dev_run
        assert trainer.checkpoint_callback == checkpoint_callback
        assert not os.path.exists(checkpoint_callback.dirpath)
        assert len(trainer.dev_debugger.checkpoint_callback_history) == 0

        # early stopping should not have been called with fast_dev_run
        assert trainer.early_stopping_callback == early_stopping_callback
        assert len(trainer.dev_debugger.early_stopping_history) == 0

    train_val_step_model = FastDevRunModel()
    trainer = Trainer(**trainer_config)
    results = trainer.fit(train_val_step_model)
    assert results

    # make sure both training_step and validation_step were called
    assert train_val_step_model.training_step_called
    assert train_val_step_model.validation_step_called

    _make_fast_dev_run_assertions(trainer)

    # -----------------------
    # also called once with no val step
    # -----------------------
    train_step_only_model = FastDevRunModel()
    train_step_only_model.validation_step = None

    trainer = Trainer(**trainer_config)
    results = trainer.fit(train_step_only_model)
    assert results

    # make sure only training_step was called
    assert train_step_only_model.training_step_called
    assert not train_step_only_model.validation_step_called

    _make_fast_dev_run_assertions(trainer)
