import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.base import DummyLogger
from tests.helpers import BoringModel


@pytest.mark.parametrize('tuner_alg', ['batch size scaler', 'learning rate finder'])
def test_skip_on_fast_dev_run_tuner(tmpdir, tuner_alg):
    """ Test that tuner algorithms are skipped if fast dev run is enabled """

    model = BoringModel()
    model.lr = 0.1  # avoid no-lr-found exception
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_scale_batch_size=(tuner_alg == 'batch size scaler'),
        auto_lr_find=(tuner_alg == 'learning rate finder'),
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
            self.training_step_call_count = 0
            self.training_epoch_end_call_count = 0
            self.validation_step_call_count = 0
            self.validation_epoch_end_call_count = 0
            self.test_step_call_count = 0

        def training_step(self, batch, batch_idx):
            self.log('some_metric', torch.tensor(7.))
            self.logger.experiment.dummy_log('some_distribution', torch.randn(7) + batch_idx)
            self.training_step_call_count += 1
            return super().training_step(batch, batch_idx)

        def training_epoch_end(self, outputs):
            self.training_epoch_end_call_count += 1
            super().training_epoch_end(outputs)

        def validation_step(self, batch, batch_idx):
            self.validation_step_call_count += 1
            return super().validation_step(batch, batch_idx)

        def validation_epoch_end(self, outputs):
            self.validation_epoch_end_call_count += 1
            super().validation_epoch_end(outputs)

        def test_step(self, batch, batch_idx):
            self.test_step_call_count += 1
            return super().test_step(batch, batch_idx)

    checkpoint_callback = ModelCheckpoint()
    early_stopping_callback = EarlyStopping()
    trainer_config = dict(
        default_root_dir=tmpdir,
        fast_dev_run=fast_dev_run,
        val_check_interval=2,
        logger=True,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    def _make_fast_dev_run_assertions(trainer, model):
        # check the call count for train/val/test step/epoch
        assert model.training_step_call_count == fast_dev_run
        assert model.training_epoch_end_call_count == 1
        assert model.validation_step_call_count == 0 if model.validation_step is None else fast_dev_run
        assert model.validation_epoch_end_call_count == 0 if model.validation_step is None else 1
        assert model.test_step_call_count == fast_dev_run

        # check trainer arguments
        assert trainer.max_steps == fast_dev_run
        assert trainer.num_sanity_val_steps == 0
        assert trainer.max_epochs == 1
        assert trainer.val_check_interval == 1.0
        assert trainer.check_val_every_n_epoch == 1

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
    trainer.fit(train_val_step_model)
    trainer.test(ckpt_path=None)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    _make_fast_dev_run_assertions(trainer, train_val_step_model)

    # -----------------------
    # also called once with no val step
    # -----------------------
    train_step_only_model = FastDevRunModel()
    train_step_only_model.validation_step = None

    trainer = Trainer(**trainer_config)
    trainer.fit(train_step_only_model)
    trainer.test(ckpt_path=None)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    _make_fast_dev_run_assertions(trainer, train_step_only_model)
