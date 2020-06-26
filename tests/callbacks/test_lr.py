import pytest

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger
from tests.base import EvalModelTemplate


def test_lr_logger_single_lr(tmpdir):
    """ Test that learning rates are extracted and logged for single lr scheduler. """
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__single_scheduler

    lr_logger = LearningRateLogger()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_logger],
    )
    result = trainer.fit(model)
    assert result

    assert lr_logger.lrs, 'No learning rates logged'
    assert len(lr_logger.lrs) == len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of lr schedulers'
    assert all([k in ['lr-Adam'] for k in lr_logger.lrs.keys()]), \
        'Names of learning rates not set correctly'


def test_lr_logger_no_lr(tmpdir):
    tutils.reset_seed()

    model = EvalModelTemplate()

    lr_logger = LearningRateLogger()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_logger],
    )

    with pytest.warns(RuntimeWarning):
        result = trainer.fit(model)
        assert result


def test_lr_logger_multi_lrs(tmpdir):
    """ Test that learning rates are extracted and logged for multi lr schedulers. """
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__multiple_schedulers

    lr_logger = LearningRateLogger()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_logger],
    )
    result = trainer.fit(model)
    assert result

    assert lr_logger.lrs, 'No learning rates logged'
    assert len(lr_logger.lrs) == len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of lr schedulers'
    assert all([k in ['lr-Adam', 'lr-Adam-1'] for k in lr_logger.lrs.keys()]), \
        'Names of learning rates not set correctly'
    assert all(len(lr) == trainer.max_epochs for k, lr in lr_logger.lrs.items()), \
        'Length of logged learning rates exceeds the number of epochs'


def test_lr_logger_param_groups(tmpdir):
    """ Test that learning rates are extracted and logged for single lr scheduler. """
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__param_groups

    lr_logger = LearningRateLogger()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_val_batches=0.1,
        limit_train_batches=0.5,
        callbacks=[lr_logger],
    )
    result = trainer.fit(model)
    assert result

    assert lr_logger.lrs, 'No learning rates logged'
    assert len(lr_logger.lrs) == 2 * len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of param groups'
    assert all([k in ['lr-Adam/pg1', 'lr-Adam/pg2'] for k in lr_logger.lrs.keys()]), \
        'Names of learning rates not set correctly'
