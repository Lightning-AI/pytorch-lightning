# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import os
import pickle
import platform
from unittest import mock
from unittest.mock import ANY, call

import pytest

import tests.base.develop_utils as tutils
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import (
    CometLogger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    TestTubeLogger,
    WandbLogger,
)
from pytorch_lightning.loggers.base import DummyExperiment
from tests.base import BoringModel, EvalModelTemplate
from tests.loggers.test_comet import _patch_comet_atexit
from tests.loggers.test_mlflow import mock_mlflow_run_creation


def _get_logger_args(logger_class, save_dir):
    logger_args = {}
    if 'save_dir' in inspect.getfullargspec(logger_class).args:
        logger_args.update(save_dir=str(save_dir))
    if 'offline_mode' in inspect.getfullargspec(logger_class).args:
        logger_args.update(offline_mode=True)
    if 'offline' in inspect.getfullargspec(logger_class).args:
        logger_args.update(offline=True)
    return logger_args


def _instantiate_logger(logger_class, save_idr, **override_kwargs):
    args = _get_logger_args(logger_class, save_idr)
    args.update(**override_kwargs)
    logger = logger_class(**args)
    return logger


def test_loggers_fit_test_all(tmpdir, monkeypatch):
    """ Verify that basic functionality of all loggers. """

    _test_loggers_fit_test(tmpdir, TensorBoardLogger)

    with mock.patch('pytorch_lightning.loggers.comet.comet_ml'), \
         mock.patch('pytorch_lightning.loggers.comet.CometOfflineExperiment'):
        _patch_comet_atexit(monkeypatch)
        _test_loggers_fit_test(tmpdir, CometLogger)

    with mock.patch('pytorch_lightning.loggers.mlflow.mlflow'), \
         mock.patch('pytorch_lightning.loggers.mlflow.MlflowClient'):
        _test_loggers_fit_test(tmpdir, MLFlowLogger)

    with mock.patch('pytorch_lightning.loggers.neptune.neptune'):
        _test_loggers_fit_test(tmpdir, NeptuneLogger)

    with mock.patch('pytorch_lightning.loggers.test_tube.Experiment'):
        _test_loggers_fit_test(tmpdir, TestTubeLogger)

    with mock.patch('pytorch_lightning.loggers.wandb.wandb') as wandb:
        wandb.run = None
        wandb.init().step = 0
        _test_loggers_fit_test(tmpdir, WandbLogger)


def _test_loggers_fit_test(tmpdir, logger_class):
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

    if logger_class == CometLogger:
        logger.experiment.id = 'foo'
        logger.experiment.project_name = 'bar'

    if logger_class == TestTubeLogger:
        logger.experiment.version = 'foo'
        logger.experiment.name = 'bar'

    if logger_class == MLFlowLogger:
        logger = mock_mlflow_run_creation(logger, experiment_id="foo", run_id="bar")

    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        default_root_dir=tmpdir,
    )
    trainer.fit(model)
    trainer.test()

    log_metric_names = [(s, sorted(m.keys())) for s, m in logger.history]
    if logger_class == TensorBoardLogger:
        expected = [
            (0, ['hp_metric']),
            (0, ['epoch', 'train_some_val']),
            (0, ['early_stop_on', 'epoch', 'val_acc']),
            (0, ['hp_metric']),
            (1, ['epoch', 'test_acc', 'test_loss'])
        ]
        assert log_metric_names == expected
    else:
        expected = [
            (0, ['epoch', 'train_some_val']),
            (0, ['early_stop_on', 'epoch', 'val_acc']),
            (1, ['epoch', 'test_acc', 'test_loss'])
        ]
        assert log_metric_names == expected


def test_loggers_save_dir_and_weights_save_path_all(tmpdir, monkeypatch):
    """ Test the combinations of save_dir, weights_save_path and default_root_dir. """

    _test_loggers_save_dir_and_weights_save_path(tmpdir, TensorBoardLogger)

    with mock.patch('pytorch_lightning.loggers.comet.comet_ml'), \
         mock.patch('pytorch_lightning.loggers.comet.CometOfflineExperiment'):
        _patch_comet_atexit(monkeypatch)
        _test_loggers_save_dir_and_weights_save_path(tmpdir, CometLogger)

    with mock.patch('pytorch_lightning.loggers.mlflow.mlflow'), \
         mock.patch('pytorch_lightning.loggers.mlflow.MlflowClient'):
        _test_loggers_save_dir_and_weights_save_path(tmpdir, MLFlowLogger)

    with mock.patch('pytorch_lightning.loggers.test_tube.Experiment'):
        _test_loggers_save_dir_and_weights_save_path(tmpdir, TestTubeLogger)

    with mock.patch('pytorch_lightning.loggers.wandb.wandb'):
        _test_loggers_save_dir_and_weights_save_path(tmpdir, WandbLogger)


def _test_loggers_save_dir_and_weights_save_path(tmpdir, logger_class):

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
    CometLogger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    TestTubeLogger,
    # The WandbLogger gets tested for pickling in its own test.
])
def test_loggers_pickle_all(tmpdir, monkeypatch, logger_class):
    """ Test that the logger objects can be pickled. This test only makes sense if the packages are installed. """
    _patch_comet_atexit(monkeypatch)
    try:
        _test_loggers_pickle(tmpdir, monkeypatch, logger_class)
    except (ImportError, ModuleNotFoundError):
        pytest.xfail(f"pickle test requires {logger_class.__class__} dependencies to be installed.")


def _test_loggers_pickle(tmpdir, monkeypatch, logger_class):
    """Verify that pickling trainer with logger works."""
    _patch_comet_atexit(monkeypatch)

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
    trainer.tune(model)
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


@pytest.mark.parametrize("logger_class", [
    CometLogger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    TestTubeLogger,
])
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_logger_created_on_rank_zero_only(tmpdir, monkeypatch, logger_class):
    """ Test that loggers get replaced by dummy loggers on global rank > 0"""
    _patch_comet_atexit(monkeypatch)
    try:
        _test_logger_created_on_rank_zero_only(tmpdir, logger_class)
    except (ImportError, ModuleNotFoundError):
        pytest.xfail(f"multi-process test requires {logger_class.__class__} dependencies to be installed.")


def _test_logger_created_on_rank_zero_only(tmpdir, logger_class):
    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = logger_class(**logger_args)
    model = EvalModelTemplate()
    trainer = Trainer(
        logger=logger,
        default_root_dir=tmpdir,
        accelerator='ddp_cpu',
        num_processes=2,
        max_steps=1,
        checkpoint_callback=True,
        callbacks=[RankZeroLoggerCheck()],
    )
    result = trainer.fit(model)
    assert result == 1


def test_logger_with_prefix_all(tmpdir, monkeypatch):
    """
    Test that prefix is added at the beginning of the metric keys.
    """
    prefix = 'tmp'

    # Comet
    with mock.patch('pytorch_lightning.loggers.comet.comet_ml'), \
         mock.patch('pytorch_lightning.loggers.comet.CometOfflineExperiment'):
        _patch_comet_atexit(monkeypatch)
        logger = _instantiate_logger(CometLogger, save_idr=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log_metrics.assert_called_once_with({"tmp-test": 1.0}, epoch=None, step=0)

    # MLflow
    with mock.patch('pytorch_lightning.loggers.mlflow.mlflow'), \
         mock.patch('pytorch_lightning.loggers.mlflow.MlflowClient'):
        logger = _instantiate_logger(MLFlowLogger, save_idr=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log_metric.assert_called_once_with(ANY, "tmp-test", 1.0, ANY, 0)

    # Neptune
    with mock.patch('pytorch_lightning.loggers.neptune.neptune'):
        logger = _instantiate_logger(NeptuneLogger, save_idr=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log_metric.assert_called_once_with("tmp-test", x=0, y=1.0)

    # TensorBoard
    with mock.patch('pytorch_lightning.loggers.tensorboard.SummaryWriter'):
        logger = _instantiate_logger(TensorBoardLogger, save_idr=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.add_scalar.assert_called_once_with("tmp-test", 1.0, 0)

    # TestTube
    with mock.patch('pytorch_lightning.loggers.test_tube.Experiment'):
        logger = _instantiate_logger(TestTubeLogger, save_idr=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log.assert_called_once_with({"tmp-test": 1.0}, global_step=0)

    # WandB
    with mock.patch('pytorch_lightning.loggers.wandb.wandb') as wandb:
        logger = _instantiate_logger(WandbLogger, save_idr=tmpdir, prefix=prefix)
        wandb.run = None
        wandb.init().step = 0
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log.assert_called_once_with({'tmp-test': 1.0}, step=0)
