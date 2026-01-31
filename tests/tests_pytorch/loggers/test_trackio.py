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
from unittest import mock

import pytest

from lightning.pytorch.loggers import TrackioLogger


def test_trackio_logger_not_available():
    with mock.patch("lightning.pytorch.loggers.trackio._TRACKIO_AVAILABLE", False), pytest.raises(ModuleNotFoundError):
        TrackioLogger(project="test")


def test_trackio_logger_init(trackio_mock):
    from trackio import init

    logger = TrackioLogger(project="test_project", name="test_name", resume="never")

    assert logger.name == "test_project"
    assert logger.version == "test_name"

    _ = logger.experiment
    _ = logger.experiment  # call a second time to test caching

    init.assert_called_once_with(
        project="test_project",
        name="test_name",
        resume="never",
    )


def test_trackio_logger_log_hyperparams(trackio_mock):
    logger = TrackioLogger(project="test_project")
    hparams = {"lr": 0.001, "batch_size": 32}
    logger.log_hyperparams(hparams)
    assert logger.experiment.config["lr"] == 0.001
    assert logger.experiment.config["batch_size"] == 32


def test_trackio_logger_log_metrics(trackio_mock):
    logger = TrackioLogger(project="test_project")
    metrics = {"train_loss": 0.5, "val_loss": 0.4}
    step = 10
    logger.log_metrics(metrics, step=step)
    logger.experiment.log.assert_called_once_with(metrics, step=step)
