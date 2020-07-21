"""
Tests to ensure that the training loop works with a dict
"""
import os
import torch
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
from pytorch_lightning.core.step_result import Result, TrainResult, EvalResult
from tests.base import EvalModelTemplate
import pytest


# test with train_step_end
# add logging + row interval tests

def test_val_step_result_callbacks(tmpdir):
    """
    Tests that val step can be used:
    - val step
    - no other val_xxx
    - train loop
    - callbacks coming from val loop (not train loop)
    """
    # enable internal debugging actions
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_log_epoch_and_step_for_callbacks
    model.training_step_end = None
    model.training_epoch_end = None
    model.validation_step = model.validation_step_result_callbacks
    model.validation_step_end = None
    model.validation_epoch_end = None

    batches = 3
    epochs = 300
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        early_stop_callback=True,
        row_log_interval=1,
        limit_train_batches=batches,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called

    # assert that early stopping happened after the requested num of steps
    # if it used the train step for ES then it wouldn't be 5
    assert len(trainer.dev_debugger.early_stopping_history) == 5

    # only 2 checkpoints expected
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 2

    # make sure the last known metric is correct
    assert trainer.callback_metrics['val_checkpoint_on'] == 171 + 22

    # did not request any metrics to log (except the metrics saying which epoch we are on)
    assert len(trainer.progress_bar_metrics) == 0
    assert len(trainer.dev_debugger.logged_metrics) == 5


def test_val_step_using_train_callbacks(tmpdir):
    """
    ES conditioned in train
    CKPT conditioned in val
    """
    # enable internal debugging actions
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_log_epoch_and_step_for_callbacks
    model.training_step_end = None
    model.training_epoch_end = None
    model.validation_step = model.validation_step_result_no_callbacks
    model.validation_step_end = None
    model.validation_epoch_end = None

    batches = 3
    epochs = 300
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        early_stop_callback=True,
        row_log_interval=1,
        limit_train_batches=batches,
        weights_summary=None,
    )
    trainer.fit(model)

    expected_epochs = 10

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called

    # early stopping was not conditioned in val loop, but instead in train loop
    assert len(trainer.dev_debugger.early_stopping_history) == expected_epochs

    # only 2 checkpoints expected
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 2

    # make sure the last known metric is correct
    assert trainer.callback_metrics['val_checkpoint_on'] == 171 + 50

    # did not request any metrics to log (except the metrics saying which epoch we are on)
    assert len(trainer.progress_bar_metrics) == 0
    assert len(trainer.dev_debugger.logged_metrics) == expected_epochs


def test_val_step_only_metrics(tmpdir):
    """
    Make sure the logged + pbar metrics are allocated accordingly
    """
    # TODO: Log and pbar metrics

    # enable internal debugging actions
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_log_epoch_and_step_for_callbacks
    model.training_step_end = None
    model.training_epoch_end = None
    model.validation_step = model.validation_step_result_only_metrics
    model.validation_step_end = None
    model.validation_epoch_end = None

    batches = 3
    epochs = 300
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        early_stop_callback=True,
        row_log_interval=1,
        limit_train_batches=batches,
        weights_summary=None,
    )
    trainer.fit(model)

    expected_epochs = 10

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called

    # early stopping was not conditioned in val loop, but instead in train loop
    assert len(trainer.dev_debugger.early_stopping_history) == expected_epochs

    # only 2 checkpoints expected
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 2

    # make sure the last known metric is correct
    assert trainer.callback_metrics['val_checkpoint_on'] == 171 + 50

    # did not request any metrics to log (except the metrics saying which epoch we are on)
    assert len(trainer.progress_bar_metrics) == 0
    assert len(trainer.dev_debugger.logged_metrics) == expected_epochs


test_val_step_only_metrics('')