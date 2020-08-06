import atexit
import inspect
import os
import pickle
import platform
from unittest import mock

import pytest

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import (
    TensorBoardLogger,
    MLFlowLogger,
    NeptuneLogger,
    TestTubeLogger,
    CometLogger,
    WandbLogger,
)
from pytorch_lightning.loggers.base import DummyExperiment
from tests.base import EvalModelTemplate


def _get_logger_args(logger_class, save_dir):
    logger_args = {}
    if 'save_dir' in inspect.getfullargspec(logger_class).args:
        logger_args.update(save_dir=str(save_dir))
    if 'offline_mode' in inspect.getfullargspec(logger_class).args:
        logger_args.update(offline_mode=True)
    if 'offline' in inspect.getfullargspec(logger_class).args:
        logger_args.update(offline=True)
    return logger_args


@pytest.mark.parametrize("logger_class", [
    TensorBoardLogger,
    CometLogger,
    MLFlowLogger,
    NeptuneLogger,
    TestTubeLogger,
    WandbLogger,
])
@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_loggers_fit_test(wandb, tmpdir, monkeypatch, logger_class):
    """Verify that basic functionality of all loggers."""
    if logger_class == CometLogger:
        # prevent comet logger from trying to print at exit, since
        # pytest's stdout/stderr redirection breaks it
        monkeypatch.setattr(atexit, 'register', lambda _: None)

    model = EvalModelTemplate()

    class StoreHistoryLogger(logger_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.history = []

        def log_metrics(self, metrics, step):
            super().log_metrics(metrics, step)
            self.history.append((step, metrics))

    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = StoreHistoryLogger(**logger_args)

    if logger_class == WandbLogger:
        # required mocks for Trainer
        logger.experiment.id = 'foo'
        logger.experiment.project_name.return_value = 'bar'

    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        limit_train_batches=0.2,
        limit_val_batches=0.5,
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    trainer.fit(model)
    trainer.test()

    log_metric_names = [(s, sorted(m.keys())) for s, m in logger.history]
    assert log_metric_names == [(0, ['epoch', 'val_acc', 'val_loss']),
                                (0, ['epoch', 'train_some_val']),
                                (1, ['epoch', 'test_acc', 'test_loss'])]


@pytest.mark.parametrize("logger_class", [
    TensorBoardLogger,
    CometLogger,
    MLFlowLogger,
    TestTubeLogger,
    WandbLogger,
])
@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_loggers_save_dir_and_weights_save_path(wandb, tmpdir, monkeypatch, logger_class):
    """ Test the combinations of save_dir, weights_save_path and default_root_dir.  """
    if logger_class == CometLogger:
        # prevent comet logger from trying to print at exit, since
        # pytest's stdout/stderr redirection breaks it
        monkeypatch.setattr(atexit, 'register', lambda _: None)

    class TestLogger(logger_class):
        # for this test it does not matter what these attributes are
        # so we standardize them to make testing easier
        @property
        def version(self):
            return 'version'

        @property
        def name(self):
            return 'name'

    model = EvalModelTemplate()
    trainer_args = dict(
        default_root_dir=tmpdir,
        max_steps=1,
    )

    # no weights_save_path given
    save_dir = tmpdir / 'logs'
    weights_save_path = None
    logger = TestLogger(**_get_logger_args(TestLogger, save_dir))
    trainer = Trainer(**trainer_args, logger=logger, weights_save_path=weights_save_path)
    trainer.fit(model)
    assert trainer.weights_save_path == trainer.default_root_dir
    assert trainer.checkpoint_callback.dirpath == os.path.join(logger.save_dir, 'name', 'version', 'checkpoints')
    assert trainer.default_root_dir == tmpdir

    # with weights_save_path given, the logger path and checkpoint path should be different
    save_dir = tmpdir / 'logs'
    weights_save_path = tmpdir / 'weights'
    logger = TestLogger(**_get_logger_args(TestLogger, save_dir))
    trainer = Trainer(**trainer_args, logger=logger, weights_save_path=weights_save_path)
    trainer.fit(model)
    assert trainer.weights_save_path == weights_save_path
    assert trainer.logger.save_dir == save_dir
    assert trainer.checkpoint_callback.dirpath == weights_save_path / 'name' / 'version' / 'checkpoints'
    assert trainer.default_root_dir == tmpdir

    # no logger given
    weights_save_path = tmpdir / 'weights'
    trainer = Trainer(**trainer_args, logger=False, weights_save_path=weights_save_path)
    trainer.fit(model)
    assert trainer.weights_save_path == weights_save_path
    assert trainer.checkpoint_callback.dirpath == weights_save_path / 'checkpoints'
    assert trainer.default_root_dir == tmpdir


@pytest.mark.parametrize("logger_class", [
    TensorBoardLogger,
    CometLogger,
    MLFlowLogger,
    NeptuneLogger,
    TestTubeLogger,
    # The WandbLogger gets tested for pickling in its own test.
])
def test_loggers_pickle(tmpdir, monkeypatch, logger_class):
    """Verify that pickling trainer with logger works."""
    if logger_class == CometLogger:
        # prevent comet logger from trying to print at exit, since
        # pytest's stdout/stderr redirection breaks it
        monkeypatch.setattr(atexit, 'register', lambda _: None)

    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = logger_class(**logger_args)

    # this can cause pickle error if the experiment object is not picklable
    # the logger needs to remove it from the state before pickle
    _ = logger.experiment

    # test pickling loggers
    pickle.dumps(logger)

    trainer = Trainer(
        max_epochs=1,
        logger=logger,
    )
    pkl_bytes = pickle.dumps(trainer)

    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({'acc': 1.0})

    # make sure we restord properly
    assert trainer2.logger.name == logger.name
    assert trainer2.logger.save_dir == logger.save_dir


@pytest.mark.parametrize("extra_params", [
    pytest.param(dict(max_epochs=1, auto_scale_batch_size=True), id='Batch-size-Finder'),
    pytest.param(dict(max_epochs=3, auto_lr_find=True), id='LR-Finder'),
])
def test_logger_reset_correctly(tmpdir, extra_params):
    """ Test that the tuners do not alter the logger reference """
    tutils.reset_seed()

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        **extra_params,
    )
    logger1 = trainer.logger
    trainer.fit(model)
    logger2 = trainer.logger
    logger3 = model.logger

    assert logger1 == logger2, \
        'Finder altered the logger of trainer'
    assert logger2 == logger3, \
        'Finder altered the logger of model'


class RankZeroLoggerCheck(Callback):
    # this class has to be defined outside the test function, otherwise we get pickle error
    # due to the way ddp process is launched

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        is_dummy = isinstance(trainer.logger.experiment, DummyExperiment)
        if trainer.is_global_zero:
            assert not is_dummy
        else:
            assert is_dummy
            assert pl_module.logger.experiment.something(foo="bar") is None


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
@pytest.mark.parametrize("logger_class", [
    TensorBoardLogger,
    CometLogger,
    MLFlowLogger,
    NeptuneLogger,
    TestTubeLogger,
    WandbLogger,
])
def test_logger_created_on_rank_zero_only(tmpdir, monkeypatch, logger_class):
    """ Test that loggers get replaced by dummy logges on global rank > 0"""
    if logger_class == CometLogger:
        # prevent comet logger from trying to print at exit, since
        # pytest's stdout/stderr redirection breaks it
        monkeypatch.setattr(atexit, 'register', lambda _: None)

    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = logger_class(**logger_args)
    model = EvalModelTemplate()
    trainer = Trainer(
        logger=logger,
        default_root_dir=tmpdir,
        distributed_backend='ddp_cpu',
        num_processes=2,
        max_steps=1,
        checkpoint_callback=True,
        callbacks=[RankZeroLoggerCheck()],
    )
    result = trainer.fit(model)
    assert result == 1
