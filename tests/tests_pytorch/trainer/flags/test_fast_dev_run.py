import os
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.pytorch.tuner.tuning import Tuner


def test_skip_on_fast_dev_run_tuner(tmp_path):
    """Test that tuner algorithms are skipped if fast dev run is enabled."""
    model = BoringModel()
    model.lr = 0.1  # avoid no-lr-found exception
    model.batch_size = 8
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        fast_dev_run=True,
    )
    tuner = Tuner(trainer)

    with pytest.warns(UserWarning, match="Skipping learning rate finder since `fast_dev_run` is enabled."):
        tuner.lr_find(model)

    with pytest.warns(UserWarning, match="Skipping batch size scaler since `fast_dev_run` is enabled."):
        tuner.scale_batch_size(model)


@pytest.mark.parametrize("fast_dev_run", [1, 4])
def test_callbacks_and_logger_not_called_with_fastdevrun(tmp_path, fast_dev_run):
    """Test that ModelCheckpoint, EarlyStopping and Logger are turned off with fast_dev_run."""

    class FastDevRunModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.training_step_call_count = 0
            self.on_train_epoch_end_call_count = 0
            self.validation_step_call_count = 0
            self.on_validation_epoch_end_call_count = 0
            self.test_step_call_count = 0

        def training_step(self, batch, batch_idx):
            self.log("some_metric", torch.tensor(7.0))
            self.logger.experiment.dummy_log("some_distribution", torch.randn(7) + batch_idx)
            self.training_step_call_count += 1
            return super().training_step(batch, batch_idx)

        def on_train_epoch_end(self):
            self.on_train_epoch_end_call_count += 1

        def validation_step(self, batch, batch_idx):
            self.validation_step_call_count += 1
            return super().validation_step(batch, batch_idx)

        def on_validation_epoch_end(self):
            self.on_validation_epoch_end_call_count += 1

        def test_step(self, batch, batch_idx):
            self.test_step_call_count += 1
            return super().test_step(batch, batch_idx)

    checkpoint_callback = ModelCheckpoint()
    checkpoint_callback.save_checkpoint = Mock()
    early_stopping_callback = EarlyStopping(monitor="foo")
    early_stopping_callback._evaluate_stopping_criteria = Mock()
    trainer_config = {
        "default_root_dir": tmp_path,
        "fast_dev_run": fast_dev_run,
        "val_check_interval": 2,
        "logger": TensorBoardLogger(tmp_path),
        "log_every_n_steps": 1,
        "callbacks": [checkpoint_callback, early_stopping_callback],
    }

    def _make_fast_dev_run_assertions(trainer, model):
        # check the call count for train/val/test step/epoch
        assert model.training_step_call_count == fast_dev_run
        assert model.on_train_epoch_end_call_count == 1
        assert model.validation_step_call_count == 0 if model.validation_step is None else fast_dev_run
        assert model.on_validation_epoch_end_call_count == 0 if model.validation_step is None else 1
        assert model.test_step_call_count == fast_dev_run

        # check trainer arguments
        assert trainer.max_steps == fast_dev_run
        assert trainer.num_sanity_val_steps == 0
        assert trainer.max_epochs == 1
        assert trainer.val_check_interval == 1.0
        assert trainer.check_val_every_n_epoch == 1

        # there should be no logger with fast_dev_run
        assert isinstance(trainer.logger, DummyLogger)

        # checkpoint callback should not have been called with fast_dev_run
        assert trainer.checkpoint_callback == checkpoint_callback
        checkpoint_callback.save_checkpoint.assert_not_called()
        assert not os.path.exists(checkpoint_callback.dirpath)

        # early stopping should not have been called with fast_dev_run
        assert trainer.early_stopping_callback == early_stopping_callback
        early_stopping_callback._evaluate_stopping_criteria.assert_not_called()

    train_val_step_model = FastDevRunModel()
    trainer = Trainer(**trainer_config)
    trainer.fit(train_val_step_model)
    trainer.test(train_val_step_model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    _make_fast_dev_run_assertions(trainer, train_val_step_model)

    # -----------------------
    # also called once with no val step
    # -----------------------
    train_step_only_model = FastDevRunModel()
    train_step_only_model.validation_step = None

    trainer = Trainer(**trainer_config)
    trainer.fit(train_step_only_model)
    trainer.test(train_step_only_model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    _make_fast_dev_run_assertions(trainer, train_step_only_model)
