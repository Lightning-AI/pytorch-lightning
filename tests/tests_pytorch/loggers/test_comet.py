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
import os
from unittest import mock
from unittest.mock import Mock, call

from lightning.pytorch.loggers import CometLogger
from torch import tensor


def _patch_comet_atexit(monkeypatch):
    """Prevent comet logger from trying to print at exit, since pytest's stdout/stderr redirection breaks it."""
    import atexit

    monkeypatch.setattr(atexit, "register", lambda _: None)


@mock.patch.dict(os.environ, {})
def test_comet_logger_online(comet_mock):
    """Test comet online with mocks."""

    comet_start = comet_mock.start

    # Test api_key given with old param "project_name"
    _logger = CometLogger(api_key="key", workspace="dummy-test", project_name="general")
    comet_start.assert_called_once_with(
        api_key="key",
        workspace="dummy-test",
        project="general",
        experiment_key=None,
        mode=None,
        online=None,
        experiment_config=comet_mock.ExperimentConfig(),
    )

    # Test online given
    comet_start.reset_mock()
    _logger = CometLogger(save_dir="test", api_key="key", workspace="dummy-test", project_name="general", online=True)
    comet_start.assert_called_once_with(
        api_key="key",
        workspace="dummy-test",
        project="general",
        experiment_key=None,
        mode=None,
        online=True,
        experiment_config=comet_mock.ExperimentConfig(),
    )

    # Test experiment_key given
    comet_start.reset_mock()
    _logger = CometLogger(
        experiment_key="test_key",
        api_key="key",
        project="general",
    )
    comet_start.assert_called_once_with(
        api_key="key",
        workspace=None,
        project="general",
        experiment_key="test_key",
        mode=None,
        online=None,
        experiment_config=comet_mock.ExperimentConfig(),
    )


@mock.patch.dict(os.environ, {})
def test_comet_experiment_is_still_alive_after_training_complete(comet_mock):
    """Test that the CometLogger will not end an experiment after training is complete."""

    logger = CometLogger()
    assert logger.experiment is not None

    logger._experiment = Mock()
    logger.finalize("ended")

    # Assert that data was saved to comet.com
    logger._experiment.flush.assert_called_once()

    # Assert that was not ended
    logger._experiment.end.assert_not_called()


@mock.patch.dict(os.environ, {})
def test_comet_logger_experiment_name(comet_mock):
    """Test that Comet Logger experiment name works correctly."""
    api_key = "api_key"
    experiment_name = "My Experiment Name"

    comet_start = comet_mock.start

    logger = CometLogger(api_key=api_key, experiment_name=experiment_name)
    comet_start.assert_called_once_with(
        api_key=api_key,
        workspace=None,
        project=None,
        experiment_key=None,
        mode=None,
        online=None,
        experiment_config=comet_mock.ExperimentConfig(),
    )
    # check that we saved "experiment name" in kwargs
    assert logger._kwargs["experiment_name"] == experiment_name

    # check that "experiment name" was passed to experiment config
    assert call(experiment_name=experiment_name) in comet_mock.ExperimentConfig.call_args_list


@mock.patch.dict(os.environ, {})
def test_comet_version(comet_mock):
    """Test that CometLogger.version returns an Experiment key."""
    api_key = "key"
    experiment_name = "My Name"

    logger = CometLogger(api_key=api_key, experiment_name=experiment_name)
    assert logger._experiment is not None
    _ = logger.version

    logger._experiment.get_key.assert_called()


@mock.patch.dict(os.environ, {})
def test_comet_epoch_logging(comet_mock, tmp_path, monkeypatch):
    """Test that CometLogger removes the epoch key from the metrics dict and passes it as argument."""
    _patch_comet_atexit(monkeypatch)
    logger = CometLogger(project_name="test", save_dir=str(tmp_path))
    logger.log_metrics({"test": 1, "epoch": 1}, step=123)
    logger.experiment.__internal_api__log_metrics__.assert_called_once_with(
        {"test": 1},
        epoch=1,
        step=123,
        prefix=logger._prefix,
        framework="pytorch-lightning",
    )


@mock.patch.dict(os.environ, {})
def test_comet_metrics_safe(comet_mock, tmp_path, monkeypatch):
    """Test that CometLogger.log_metrics doesn't do inplace modification of metrics."""
    _patch_comet_atexit(monkeypatch)
    logger = CometLogger(project_name="test", save_dir=str(tmp_path))
    metrics = {"tensor": tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), "epoch": 1}
    logger.log_metrics(metrics)
    assert metrics["tensor"].requires_grad
