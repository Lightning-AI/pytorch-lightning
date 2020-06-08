import glob
import math
import os
import types
from argparse import Namespace

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.lightning import CHECKPOINT_KEY_MODULE_ARGS
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml, save_hparams_to_tags_csv
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from tests.base.determininistic_model import DeterministicModel


def test_trainingstep_dict(tmpdir):
    """
    Tests that only training_step can be used
    """
    model_dict = DeterministicModel()
    model_dict.training_step = model_dict.training_step_dict_return
    model_dict.val_dataloader = None

    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model_dict)

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model_dict.train_dataloader()):
        break

    # TODO: split back to EvalResult and TrainResult (because of the defaults on when to log and minimize)
    out, training_step_output_for_epoch_end = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert out.to_log_on_batch_end['log_acc1'] == 12.0
    assert out.to_log_on_batch_end['log_acc2'] == 7.0
    assert out.to_log_on_batch_end['pbar_acc1'] == 17.0
    assert out.to_log_on_batch_end['pbar_acc2'] == 19.0
    assert out.early_stop_on == 1.4
    assert out.checkpoint_on == 1.5


test_trainingstep_dict('')

def test_trainingstep_result(tmpdir):
    """
    Tests that only training_step can be used
    """
    model = DeterministicModel()
    model.training_step = model.training_step_only
    model.val_dataloader = None

    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model)

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        pass

    out, training_step_output_for_epoch_end = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert out.to_log_on_batch_end['log_acc1'] == 12.0
    assert out.to_log_on_batch_end['log_acc2'] == 7.0
    assert out.to_log_on_batch_end['pbar_acc1'] == 17.0
    assert out.to_log_on_batch_end['pbar_acc2'] == 19.0
    assert out.early_stop_on == 1.4
    assert out.checkpoint_on == 1.5


test_trainingstep('')

def test_trainingstep_evalstep_result_return(tmpdir):
    """
    Verifies training_step and validation_step functionality
    when a Result is used
    """
    # ------------------
    # test EvalReturn
    # ------------------
    model = DeterministicModel()
    model.training_step = model.training_step_only
    model.validation_step = model.validation_step_only

    loaders = [model.train_dataloader()]
    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model)

    # make sure evaluate outputs what is expected
    out = trainer._evaluate(model, loaders, max_batches=2, test_mode=False)
    assert out.log_on_epoch_end['log_acc1'] == 12.0
    assert out.log_on_epoch_end['log_acc2'] == 7.0
    assert out.pbar_on_epoch_end['pbar_acc1'] == 17.0
    assert out.pbar_on_epoch_end['pbar_acc2'] == 19.0
    assert out.early_stop_on == 1.4
    assert out.checkpoint_on == 1.5


def test_trainingstep_evalstep_dict_return(tmpdir):
    """
    Verify training_step and validation_step functionality when dict is used
    """

    # Test 1:
    # dict returned. Use the metric to track as early_stopping, etc...
    model = DeterministicModel()
    model.training_step = model.training_step_dict_return
    model.validation_step = model.validation_step_dict_return

    loaders = [model.train_dataloader()]
    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model)

    out = trainer._evaluate(model, loaders, max_batches=2, test_mode=False)
    assert out.log_on_epoch_end['log_acc1'] == 12.0
    assert out.log_on_epoch_end['log_acc2'] == 7.0
    assert out.pbar_on_epoch_end['pbar_acc1'] == 17.0
    assert out.pbar_on_epoch_end['pbar_acc2'] == 19.0
    assert 'early_stop_on' not in out
    assert out.checkpoint_on == 171.0

    # Test 2:
    # when user defines early stopping, the loss returned in the dict should be used for that
    model = DeterministicModel()
    model.training_step = model.training_step_dict_return
    model.validation_step = model.validation_step_dict_return

    loaders = [model.train_dataloader()]
    trainer = Trainer(fast_dev_run=True, weights_summary=None, early_stop_callback=True)
    trainer.fit(model)

    out = trainer._evaluate(model, loaders, max_batches=2, test_mode=False)
    assert out.early_stop_on == 171.0
    assert out.checkpoint_on == 171.0


test_trainingstep_evalstep_result_return('')
test_trainingstep_evalstep_dict_return('')


def test_train_val_step_end(tmpdir):
    """
    Verifies:
    - training_step + training_step_end
    - validation_step + validation_step_end
    """
    # ------------------
    # test EvalReturn
    # ------------------
    model = DeterministicModel()
    model.training_step = model.training_step_with_batch_end
    model.training_step_end = model.training_step_end_basic
    model.validation_step = model.validation_step_with_batch_end
    model.validation_step_end = model.validation_step_end_basic

    loaders = [model.train_dataloader()]
    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model)

    # make sure evaluate outputs what is expected
    out = trainer._evaluate(model, loaders, max_batches=2, test_mode=False)
    assert out.log_on_epoch_end['log_acc1'] == 12.0
    assert out.log_on_epoch_end['log_acc2'] == 7.0
    assert out.pbar_on_epoch_end['pbar_acc1'] == 17.0
    assert out.pbar_on_epoch_end['pbar_acc2'] == 19.0
    assert out['early_stop_on'] == 1.4
    assert out['checkpoint_on'] == 1.5

    # ---------------------
    # test dic return only
    # ---------------------
    model = DeterministicModel()
    model.training_step = model.training_step_dict_return
    model.validation_step = model.validation_step_dict_return

    loaders = [model.train_dataloader()]
    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model)

    out = trainer._evaluate(model, loaders, max_batches=2, test_mode=False)
    assert out.log_on_epoch_end['log_acc1'] == 12.0
    assert out.log_on_epoch_end['log_acc2'] == 7.0
    assert out.pbar_on_epoch_end['pbar_acc1'] == 17.0
    assert out.pbar_on_epoch_end['pbar_acc2'] == 19.0
    assert out['early_stop_on'] == 171.0
    assert out['checkpoint_on'] == 171.0


test_train_val_step_end('')

