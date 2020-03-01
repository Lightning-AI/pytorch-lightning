import os
import pickle
import pytest
import torch
from unittest.mock import MagicMock

import tests.models.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    rank_zero_only,
    TensorBoardLogger,
    MLFlowLogger,
    CometLogger,
    WandbLogger,
    NeptuneLogger
)
from tests.models import LightningTestModel


def test_testtube_logger(tmpdir):
    """Verify that basic functionality of test tube logger works."""
    tutils.reset_seed()
    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    logger = tutils.get_test_tube_logger(tmpdir, False)

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, "Training failed"


def test_testtube_pickle(tmpdir):
    """Verify that pickling a trainer containing a test tube logger works."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    logger = tutils.get_test_tube_logger(tmpdir, False)
    logger.log_hyperparams(hparams)
    logger.save()

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_mlflow_logger(tmpdir):
    """Verify that basic functionality of mlflow logger works."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    mlflow_dir = os.path.join(tmpdir, "mlruns")
    logger = MLFlowLogger("test", tracking_uri=f"file:{os.sep * 2}{mlflow_dir}")

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print('result finished')
    assert result == 1, "Training failed"


def test_mlflow_pickle(tmpdir):
    """Verify that pickling trainer with mlflow logger works."""
    tutils.reset_seed()

    # hparams = tutils.get_hparams()
    # model = LightningTestModel(hparams)

    mlflow_dir = os.path.join(tmpdir, "mlruns")
    logger = MLFlowLogger("test", tracking_uri=f"file:{os.sep * 2}{mlflow_dir}")
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_comet_logger(tmpdir, monkeypatch):
    """Verify that basic functionality of Comet.ml logger works."""

    # prevent comet logger from trying to print at exit, since
    # pytest's stdout/stderr redirection breaks it
    import atexit
    monkeypatch.setattr(atexit, "register", lambda _: None)

    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    comet_dir = os.path.join(tmpdir, "cometruns")

    # We test CometLogger in offline mode with local saves
    logger = CometLogger(
        save_dir=comet_dir,
        project_name="general",
        workspace="dummy-test",
    )

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print('result finished')
    assert result == 1, "Training failed"


def test_comet_pickle(tmpdir, monkeypatch):
    """Verify that pickling trainer with comet logger works."""

    # prevent comet logger from trying to print at exit, since
    # pytest's stdout/stderr redirection breaks it
    import atexit
    monkeypatch.setattr(atexit, "register", lambda _: None)

    tutils.reset_seed()

    # hparams = tutils.get_hparams()
    # model = LightningTestModel(hparams)

    comet_dir = os.path.join(tmpdir, "cometruns")

    # We test CometLogger in offline mode with local saves
    logger = CometLogger(
        save_dir=comet_dir,
        project_name="general",
        workspace="dummy-test",
    )

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_wandb_logger(tmpdir):
    """Verify that basic functionality of wandb logger works."""
    tutils.reset_seed()

    wandb_dir = os.path.join(tmpdir, "wandb")
    _ = WandbLogger(save_dir=wandb_dir, anonymous=True, offline=True)


def test_wandb_pickle(tmpdir):
    """Verify that pickling trainer with wandb logger works."""
    tutils.reset_seed()

    wandb_dir = str(tmpdir)
    logger = WandbLogger(save_dir=wandb_dir, anonymous=True, offline=True)
    assert logger is not None


def test_neptune_logger(tmpdir):
    """Verify that basic functionality of neptune logger works."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)
    logger = NeptuneLogger(offline_mode=True)

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print('result finished')
    assert result == 1, "Training failed"


def test_neptune_pickle(tmpdir):
    """Verify that pickling trainer with neptune logger works."""
    tutils.reset_seed()

    # hparams = tutils.get_hparams()
    # model = LightningTestModel(hparams)

    logger = NeptuneLogger(offline_mode=True)

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_neptune_leave_open_experiment_after_fit(tmpdir):
    """Verify that neptune experiment was closed after training"""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    def _run_training(logger):
        logger._experiment = MagicMock()

        trainer_options = dict(
            default_save_path=tmpdir,
            max_epochs=1,
            train_percent_check=0.05,
            logger=logger
        )
        trainer = Trainer(**trainer_options)
        trainer.fit(model)
        return logger

    logger_close_after_fit = _run_training(NeptuneLogger(offline_mode=True))
    assert logger_close_after_fit._experiment.stop.call_count == 1

    logger_open_after_fit = _run_training(NeptuneLogger(offline_mode=True, close_after_fit=False))
    assert logger_open_after_fit._experiment.stop.call_count == 0



def test_tensorboard_logger(tmpdir):
    """Verify that basic functionality of Tensorboard logger works."""

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    logger = TensorBoardLogger(save_dir=tmpdir, name="tensorboard_logger_test")

    trainer_options = dict(max_epochs=1, train_percent_check=0.01, logger=logger)

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print("result finished")
    assert result == 1, "Training failed"


def test_tensorboard_pickle(tmpdir):
    """Verify that pickling trainer with Tensorboard logger works."""

    # hparams = tutils.get_hparams()
    # model = LightningTestModel(hparams)

    logger = TensorBoardLogger(save_dir=tmpdir, name="tensorboard_pickle_test")

    trainer_options = dict(max_epochs=1, logger=logger)

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_tensorboard_automatic_versioning(tmpdir):
    """Verify that automatic versioning works"""

    root_dir = tmpdir.mkdir("tb_versioning")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning")

    assert logger.version == 2


def test_tensorboard_manual_versioning(tmpdir):
    """Verify that manual versioning works"""

    root_dir = tmpdir.mkdir("tb_versioning")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")
    root_dir.mkdir("version_2")

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning", version=1)

    assert logger.version == 1


def test_tensorboard_named_version(tmpdir):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402' """

    tmpdir.mkdir("tb_versioning")
    expected_version = "2020-02-05-162402"

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning", version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2})  # Force data to be written

    assert logger.version == expected_version
    # Could also test existence of the directory but this fails in the "minimum requirements" test setup


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_metrics(tmpdir, step_idx):
    logger = TensorBoardLogger(tmpdir)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": torch.tensor(0.1),
        "IntTensor": torch.tensor(1)
    }
    logger.log_metrics(metrics, step_idx)


def test_tensorboard_log_hyperparams(tmpdir):
    logger = TensorBoardLogger(tmpdir)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True
    }
    logger.log_hyperparams(hparams)


def test_custom_logger(tmpdir):
    class CustomLogger(LightningLoggerBase):
        def __init__(self):
            super().__init__()
            self.hparams_logged = None
            self.metrics_logged = None
            self.finalized = False

        @rank_zero_only
        def log_hyperparams(self, params):
            self.hparams_logged = params

        @rank_zero_only
        def log_metrics(self, metrics, step):
            self.metrics_logged = metrics

        @rank_zero_only
        def finalize(self, status):
            self.finalized_status = status

        @property
        def name(self):
            return "name"

        @property
        def version(self):
            return "1"

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    logger = CustomLogger()

    trainer_options = dict(
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger,
        default_save_path=tmpdir
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result == 1, "Training failed"
    assert logger.hparams_logged == hparams
    assert logger.metrics_logged != {}
    assert logger.finalized_status == "success"


def test_adding_step_key(tmpdir):
    logged_step = 0

    def _validation_end(outputs):
        nonlocal logged_step
        logged_step += 1
        return {"log": {"step": logged_step, "val_acc": logged_step / 10}}

    def _log_metrics_decorator(log_metrics_fn):
        def decorated(metrics, step):
            if "val_acc" in metrics:
                assert step == logged_step
            return log_metrics_fn(metrics, step)

        return decorated

    model, hparams = tutils.get_model()
    model.validation_end = _validation_end
    trainer_options = dict(
        max_epochs=4,
        default_save_path=tmpdir,
        train_percent_check=0.001,
        val_percent_check=0.01,
        num_sanity_val_steps=0
    )
    trainer = Trainer(**trainer_options)
    trainer.logger.log_metrics = _log_metrics_decorator(trainer.logger.log_metrics)
    trainer.fit(model)
