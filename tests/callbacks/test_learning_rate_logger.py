import tests.base.utils as tutils
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateLogger
from tests.base import EvalModelTemplate


def test_lr_logger_single_lr(tmpdir):
    """ Test that learning rates are extracted and logged for single lr scheduler"""
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__single_scheduler

    lr_logger = LearningRateLogger()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        val_percent_check=0.1,
        train_percent_check=0.5,
        callbacks=[lr_logger]
    )
    results = trainer.fit(model)

    assert results == 1
    assert lr_logger.lrs, 'No learning rates logged'
    assert len(lr_logger.lrs) == len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of lr schedulers'
    assert all([k in ['lr-Adam'] for k in lr_logger.lrs.keys()]), \
        'Names of learning rates not set correctly'


def test_lr_logger_multi_lrs(tmpdir):
    """ Test that learning rates are extracted and logged for multi lr schedulers """
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__multiple_schedulers

    lr_logger = LearningRateLogger()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.5,
        callbacks=[lr_logger]
    )
    results = trainer.fit(model)

    assert results == 1
    assert lr_logger.lrs, 'No learning rates logged'
    assert len(lr_logger.lrs) == len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of lr schedulers'
    assert all([k in ['lr-Adam', 'lr-Adam-1'] for k in lr_logger.lrs.keys()]), \
        'Names of learning rates not set correctly'


def test_lr_logger_param_groups(tmpdir):
    """ Test that learning rates are extracted and logged for single lr scheduler"""
    tutils.reset_seed()

    model = EvalModelTemplate()
    model.configure_optimizers = model.configure_optimizers__param_groups

    lr_logger = LearningRateLogger()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        val_percent_check=0.1,
        train_percent_check=0.5,
        callbacks=[lr_logger]
    )
    results = trainer.fit(model)

    assert lr_logger.lrs, 'No learning rates logged'
    assert len(lr_logger.lrs) == 2 * len(trainer.lr_schedulers), \
        'Number of learning rates logged does not match number of param groups'
    assert all([k in ['lr-Adam/pg1', 'lr-Adam/pg2'] for k in lr_logger.lrs.keys()]), \
        'Names of learning rates not set correctly'
