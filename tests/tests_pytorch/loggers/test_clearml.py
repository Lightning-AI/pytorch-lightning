import os
from unittest import mock

import pytest
import torch
from clearml import Task

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.loggers.clearml import ClearMLLogger


def test_clearml_logger_log_metrics():
    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = torch.randn(1)
            loss.requires_grad = True
            self.logger.log_metrics({"loss": loss})
            return loss

    logger = ClearMLLogger("test", "test")
    model = CustomModel()
    limit_batches = 5
    max_epochs = 1
    trainer = Trainer(
        logger=logger,
        max_epochs=max_epochs,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
    )

    with mock.patch.object(logger, "log_metrics") as mock_method:
        trainer.fit(model)

    mock_method.assert_called()


def test_clearml_logger_log_hyperparams():
    logger = ClearMLLogger("test", "test")

    with mock.patch.object(logger, "log_hyperparams") as mock_method:
        logger.log_hyperparams({"lr": 1e-3})

    mock_method.assert_called()


def test_clearml_logger_log_table():
    logger = ClearMLLogger("test", "test")

    with mock.patch.object(logger, "log_table") as mock_method:
        logger.log_table("test", [[1, 2, 3], [4, 5, 6]])

    mock_method.assert_called()


def test_clearml_logger_run_name():
    logger = ClearMLLogger("test", "test")

    assert logger.name == "test"


def test_clearml_logger_reusing_task():
    logger = ClearMLLogger("test", "test")
    logger_reuse = ClearMLLogger(
        project_name="test", 
        task_name="test", 
        task_id=logger.task.id
    )

    assert logger.version == logger_reuse.version


def test_clearml_logger_run_version():
    logger = ClearMLLogger("test", "test")
    current_task = Task.current_task()

    assert logger.version == current_task.id


def test_clearml_logger_manual_step():
    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = torch.randn(1)
            loss.requires_grad = True
            self.logger.log_metrics({"loss": loss})
            return loss

    logger = ClearMLLogger("test", "test")
    model = CustomModel()
    limit_batches = 5
    max_epochs = 1
    trainer = Trainer(
        logger=logger,
        max_epochs=max_epochs,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
    )

    trainer.fit(model)
    assert logger._step == limit_batches * max_epochs


@pytest.fixture(autouse=True)
def run_around_tests():
    yield

    for env in {"CLEARML_PROC_MASTER_ID", "TRAINS_PROC_MASTER_ID"}:
        if env in os.environ:
            del os.environ[env]
