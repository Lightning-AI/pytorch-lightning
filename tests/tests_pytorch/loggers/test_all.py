# Copyright The Lightning AI team.
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
import contextlib
import inspect
import pickle
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
import torch

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import (
    CometLogger,
    CSVLogger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)
from lightning.pytorch.loggers.logger import DummyExperiment
from lightning.pytorch.loggers.tensorboard import _TENSORBOARD_AVAILABLE
from lightning.pytorch.tuner.tuning import Tuner
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.loggers.test_comet import _patch_comet_atexit
from tests_pytorch.loggers.test_mlflow import mock_mlflow_run_creation
from tests_pytorch.loggers.test_neptune import create_neptune_mock

LOGGER_CTX_MANAGERS = (
    mock.patch("lightning.pytorch.loggers.comet.comet_ml"),
    mock.patch("lightning.pytorch.loggers.comet.CometOfflineExperiment"),
    mock.patch("lightning.pytorch.loggers.mlflow._MLFLOW_AVAILABLE", return_value=True),
    mock.patch("lightning.pytorch.loggers.mlflow.MlflowClient"),
    mock.patch("lightning.pytorch.loggers.mlflow.Metric"),
    mock.patch("lightning.pytorch.loggers.neptune.neptune", new_callable=create_neptune_mock),
    mock.patch("lightning.pytorch.loggers.neptune._NEPTUNE_AVAILABLE", return_value=True),
    mock.patch("lightning.pytorch.loggers.neptune.File", new=mock.Mock()),
    mock.patch("lightning.pytorch.loggers.wandb.wandb"),
    mock.patch("lightning.pytorch.loggers.wandb.Run", new=mock.Mock),
)
ALL_LOGGER_CLASSES = (
    CometLogger,
    CSVLogger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)
ALL_LOGGER_CLASSES_WO_NEPTUNE = tuple(filter(lambda cls: cls is not NeptuneLogger, ALL_LOGGER_CLASSES))
ALL_LOGGER_CLASSES_WO_NEPTUNE_WANDB = tuple(filter(lambda cls: cls is not WandbLogger, ALL_LOGGER_CLASSES_WO_NEPTUNE))


def _get_logger_args(logger_class, save_dir):
    logger_args = {}
    if "save_dir" in inspect.getfullargspec(logger_class).args:
        logger_args.update(save_dir=str(save_dir))
    if "offline_mode" in inspect.getfullargspec(logger_class).args:
        logger_args.update(offline_mode=True)
    if "offline" in inspect.getfullargspec(logger_class).args:
        logger_args.update(offline=True)
    if issubclass(logger_class, NeptuneLogger):
        logger_args.update(mode="offline")
    return logger_args


def _instantiate_logger(logger_class, save_dir, **override_kwargs):
    args = _get_logger_args(logger_class, save_dir)
    args.update(**override_kwargs)
    logger = logger_class(**args)
    return logger


@pytest.mark.parametrize("logger_class", ALL_LOGGER_CLASSES)
def test_loggers_fit_test_all(tmpdir, monkeypatch, logger_class):
    """Verify that basic functionality of all loggers."""
    with contextlib.ExitStack() as stack:
        for mgr in LOGGER_CTX_MANAGERS:
            stack.enter_context(mgr)
        _test_loggers_fit_test(tmpdir, logger_class)


def _test_loggers_fit_test(tmpdir, logger_class):
    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("train_some_val", loss)
            return {"loss": loss}

        def on_validation_epoch_end(self):
            self.log_dict({"early_stop_on": torch.tensor(1), "val_loss": torch.tensor(0.5)})

        def on_test_epoch_end(self):
            self.log("test_loss", torch.tensor(2))

    class StoreHistoryLogger(logger_class):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.history = []

        def log_metrics(self, metrics, step):
            super().log_metrics(metrics, step)
            self.history.append((step, metrics))

    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = StoreHistoryLogger(**logger_args)

    if logger_class == WandbLogger:
        # required mocks for Trainer
        logger.experiment.id = "foo"
        logger.experiment.name = "bar"

    if logger_class == CometLogger:
        logger.experiment.id = "foo"
        logger.experiment.project_name = "bar"

    if logger_class == MLFlowLogger:
        logger = mock_mlflow_run_creation(logger, experiment_id="foo", run_id="bar")

    model = CustomModel()
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
            (0, ["epoch", "train_some_val"]),
            (0, ["early_stop_on", "epoch", "val_loss"]),
            (1, ["epoch", "test_loss"]),
        ]
        assert log_metric_names == expected
    else:
        expected = [
            (0, ["epoch", "train_some_val"]),
            (0, ["early_stop_on", "epoch", "val_loss"]),
            (1, ["epoch", "test_loss"]),
        ]
        assert log_metric_names == expected


@pytest.mark.parametrize(
    "logger_class", ALL_LOGGER_CLASSES_WO_NEPTUNE
)  # WandbLogger and NeptuneLogger get tested separately
def test_loggers_pickle_all(tmpdir, monkeypatch, logger_class):
    """Test that the logger objects can be pickled.

    This test only makes sense if the packages are installed.
    """
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

    # logger also has to avoid adding un-picklable attributes to self in .save
    logger.log_metrics({"a": 1})
    logger.save()

    # test pickling loggers
    pickle.dumps(logger)

    trainer = Trainer(max_epochs=1, logger=logger)
    pkl_bytes = pickle.dumps(trainer)

    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})

    # make sure we restored properly
    assert trainer2.logger.name == logger.name
    assert trainer2.logger.save_dir == logger.save_dir


@pytest.mark.parametrize("tuner_method", ["lr_find", "scale_batch_size"])
def test_logger_reset_correctly(tmpdir, tuner_method):
    """Test that the tuners do not alter the logger reference."""

    class CustomModel(BoringModel):
        def __init__(self, lr=0.1, batch_size=1):
            super().__init__()
            self.save_hyperparameters()

    model = CustomModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    tuner = Tuner(trainer)

    logger1 = trainer.logger
    getattr(tuner, tuner_method)(model)
    logger2 = trainer.logger
    logger3 = model.logger

    assert logger1 == logger2, "Finder altered the logger of trainer"
    assert logger2 == logger3, "Finder altered the logger of model"


class RankZeroLoggerCheck(Callback):
    # this class has to be defined outside the test function, otherwise we get pickle error
    # due to the way ddp process is launched

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        is_dummy = isinstance(trainer.logger.experiment, DummyExperiment)
        if trainer.is_global_zero:
            assert not is_dummy
        else:
            assert is_dummy
            assert pl_module.logger.experiment.something(foo="bar") is None


@pytest.mark.parametrize("logger_class", ALL_LOGGER_CLASSES_WO_NEPTUNE_WANDB)
@RunIf(skip_windows=True)
def test_logger_created_on_rank_zero_only(tmpdir, monkeypatch, logger_class):
    """Test that loggers get replaced by dummy loggers on global rank > 0."""
    _patch_comet_atexit(monkeypatch)
    try:
        _test_logger_created_on_rank_zero_only(tmpdir, logger_class)
    except (ImportError, ModuleNotFoundError):
        pytest.xfail(f"multi-process test requires {logger_class.__class__} dependencies to be installed.")


def _test_logger_created_on_rank_zero_only(tmpdir, logger_class):
    logger_args = _get_logger_args(logger_class, tmpdir)
    logger = logger_class(**logger_args)
    model = BoringModel()
    trainer = Trainer(
        logger=logger,
        default_root_dir=tmpdir,
        strategy="ddp_spawn",
        accelerator="cpu",
        devices=2,
        max_steps=1,
        callbacks=[RankZeroLoggerCheck()],
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


def test_logger_with_prefix_all(tmpdir, monkeypatch):
    """Test that prefix is added at the beginning of the metric keys."""
    prefix = "tmp"

    # Comet
    with mock.patch("lightning.pytorch.loggers.comet.comet_ml"), mock.patch(
        "lightning.pytorch.loggers.comet.CometOfflineExperiment"
    ):
        _patch_comet_atexit(monkeypatch)
        logger = _instantiate_logger(CometLogger, save_dir=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log_metrics.assert_called_once_with({"tmp-test": 1.0}, epoch=None, step=0)

    # MLflow
    with mock.patch("lightning.pytorch.loggers.mlflow._MLFLOW_AVAILABLE", return_value=True), mock.patch(
        "lightning.pytorch.loggers.mlflow.Metric"
    ) as Metric, mock.patch("lightning.pytorch.loggers.mlflow.MlflowClient"):
        logger = _instantiate_logger(MLFlowLogger, save_dir=tmpdir, prefix=prefix)
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log_batch.assert_called_once_with(
            run_id=ANY, metrics=[Metric(key="tmp-test", value=1.0, timestamp=ANY, step=0)]
        )

    # Neptune
    with mock.patch("lightning.pytorch.loggers.neptune.neptune"), mock.patch(
        "lightning.pytorch.loggers.neptune._NEPTUNE_AVAILABLE", return_value=True
    ):
        logger = _instantiate_logger(NeptuneLogger, api_key="test", project="project", save_dir=tmpdir, prefix=prefix)
        assert logger.experiment.__getitem__.call_count == 2
        logger.log_metrics({"test": 1.0}, step=0)
        assert logger.experiment.__getitem__.call_count == 3
        logger.experiment.__getitem__.assert_called_with("tmp/test")
        logger.experiment.__getitem__().log.assert_called_once_with(1.0)

    # TensorBoard
    if _TENSORBOARD_AVAILABLE:
        import torch.utils.tensorboard as tb
    else:
        import tensorboardX as tb

    monkeypatch.setattr(tb, "SummaryWriter", Mock())
    logger = _instantiate_logger(TensorBoardLogger, save_dir=tmpdir, prefix=prefix)
    logger.log_metrics({"test": 1.0}, step=0)
    logger.experiment.add_scalar.assert_called_once_with("tmp-test", 1.0, 0)

    # WandB
    with mock.patch("lightning.pytorch.loggers.wandb.wandb") as wandb, mock.patch(
        "lightning.pytorch.loggers.wandb.Run", new=mock.Mock
    ):
        logger = _instantiate_logger(WandbLogger, save_dir=tmpdir, prefix=prefix)
        wandb.run = None
        wandb.init().step = 0
        logger.log_metrics({"test": 1.0}, step=0)
        logger.experiment.log.assert_called_once_with({"tmp-test": 1.0, "trainer/global_step": 0})


def test_logger_default_name(tmpdir, monkeypatch):
    """Test that the default logger name is lightning_logs."""
    # CSV
    logger = CSVLogger(save_dir=tmpdir)
    assert logger.name == "lightning_logs"

    # TensorBoard
    if _TENSORBOARD_AVAILABLE:
        import torch.utils.tensorboard as tb
    else:
        import tensorboardX as tb

    monkeypatch.setattr(tb, "SummaryWriter", Mock())
    logger = _instantiate_logger(TensorBoardLogger, save_dir=tmpdir)
    assert logger.name == "lightning_logs"

    # MLflow
    with mock.patch("lightning.pytorch.loggers.mlflow._MLFLOW_AVAILABLE", return_value=True), mock.patch(
        "lightning.pytorch.loggers.mlflow.MlflowClient"
    ) as mlflow_client:
        mlflow_client().get_experiment_by_name.return_value = None
        logger = _instantiate_logger(MLFlowLogger, save_dir=tmpdir)

        _ = logger.experiment
        logger._mlflow_client.create_experiment.assert_called_with(name="lightning_logs", artifact_location=ANY)
        # on MLFLowLogger `name` refers to the experiment id
        # assert logger.experiment.get_experiment(logger.name).name == "lightning_logs"
