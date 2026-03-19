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
from argparse import Namespace
from unittest.mock import MagicMock, call

import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers.litlogger import LitLogger


def test_litlogger_init(litlogger_mock, tmp_path):
    """Test LitLogger initialization."""
    logger = LitLogger(
        name="test-experiment",
        root_dir=tmp_path,
        teamspace="test-teamspace",
        metadata={"key": "value"},
    )

    assert logger.name == "test-experiment"
    assert logger.root_dir == str(tmp_path)
    assert logger._teamspace == "test-teamspace"
    assert logger._metadata == {"key": "value"}


def test_litlogger_default_name(litlogger_mock, tmp_path):
    """Test LitLogger generates a name if not provided."""
    logger = LitLogger(root_dir=tmp_path)
    assert logger.name == "generated-name"


def test_litlogger_log_dir(litlogger_mock, tmp_path):
    """Test log_dir property."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    expected_log_dir = os.path.join(str(tmp_path), "test")
    assert logger.log_dir == expected_log_dir


def test_litlogger_log_dir_with_sub_dir(litlogger_mock, tmp_path):
    """Test log_dir property with sub_dir."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    logger._sub_dir = "sub"
    expected_log_dir = os.path.join(str(tmp_path), "test", "sub")
    assert logger.log_dir == expected_log_dir


def test_litlogger_save_dir(litlogger_mock, tmp_path):
    """Test save_dir property equals log_dir."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    assert logger.save_dir == logger.log_dir


def test_litlogger_experiment_property(litlogger_mock, tmp_path):
    """Test experiment property initializes litlogger."""
    logger = LitLogger(name="test", root_dir=tmp_path, teamspace="my-teamspace")
    experiment = logger.experiment

    assert experiment is not None
    litlogger_mock.Experiment.assert_called_once()

    # Check Experiment was called with correct arguments
    call_kwargs = litlogger_mock.Experiment.call_args[1]
    assert call_kwargs["name"].startswith("test-")
    assert call_kwargs["name"].endswith(logger.version)
    assert ":" not in logger.version
    assert call_kwargs["log_dir"] == os.path.join(str(tmp_path), "test")
    assert call_kwargs["teamspace"] == "my-teamspace"
    assert call_kwargs["store_step"] is True
    assert call_kwargs["store_created_at"] is True
    experiment.print_url.assert_called_once_with()


def test_litlogger_experiment_reuses_existing(litlogger_mock, tmp_path):
    """Test experiment property reuses existing experiment."""
    logger = LitLogger(name="test", root_dir=tmp_path)

    # Access experiment twice
    _ = logger.experiment
    _ = logger.experiment

    # Experiment should only be created once
    assert litlogger_mock.Experiment.call_count == 1


@pytest.mark.parametrize("step_idx", [10, None])
def test_litlogger_log_metrics(litlogger_mock, tmp_path, step_idx):
    """Test log_metrics method."""
    logger = LitLogger(name="test", root_dir=tmp_path)

    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)

    experiment = litlogger_mock.Experiment.return_value
    expected_step = 0 if step_idx is None else step_idx

    # Verify tensors are converted to Python scalars
    experiment.series_mocks["float"].append.assert_called_once_with(0.3, step=expected_step)
    experiment.series_mocks["int"].append.assert_called_once_with(1, step=expected_step)
    float_tensor_call = experiment.series_mocks["FloatTensor"].append.call_args
    int_tensor_call = experiment.series_mocks["IntTensor"].append.call_args
    assert isinstance(float_tensor_call.args[0], float)
    assert isinstance(int_tensor_call.args[0], int)
    assert float_tensor_call.kwargs == {"step": expected_step}
    assert int_tensor_call.kwargs == {"step": expected_step}


def test_litlogger_log_metrics_with_prefix(litlogger_mock, tmp_path):
    """Test log_metrics with prefix."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    logger._prefix = "train"

    logger.log_metrics({"loss": 0.5}, step=1)

    litlogger_mock.Experiment.return_value.series_mocks["train-loss"].append.assert_called_once_with(0.5, step=1)


def test_litlogger_log_hyperparams_dict(litlogger_mock, tmp_path):
    """Test log_hyperparams with dict."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    hparams = {"learning_rate": 0.001, "batch_size": 32}
    logger.log_hyperparams(hparams)

    experiment = litlogger_mock.Experiment.return_value
    assert experiment.__setitem__.mock_calls == [
        call("learning_rate", "0.001"),
        call("batch_size", "32"),
    ]


def test_litlogger_log_hyperparams_namespace(litlogger_mock, tmp_path):
    """Test log_hyperparams with Namespace."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    hparams = Namespace(learning_rate=0.001, batch_size=32)
    logger.log_hyperparams(hparams)

    experiment = litlogger_mock.Experiment.return_value
    assert experiment.__setitem__.mock_calls == [
        call("learning_rate", "0.001"),
        call("batch_size", "32"),
    ]


def test_litlogger_log_graph_warning(litlogger_mock, tmp_path):
    """Test log_graph emits warning."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    model = BoringModel()

    with pytest.warns(UserWarning, match="LitLogger does not support `log_graph`"):
        logger.log_graph(model)


def test_litlogger_finalize(litlogger_mock, tmp_path):
    """Test finalize method."""
    logger = LitLogger(name="test", root_dir=tmp_path)

    # Initialize the experiment first
    _ = logger.experiment

    logger.finalize("success")

    litlogger_mock.Experiment.return_value.finalize.assert_called_once_with("success")


def test_litlogger_finalize_no_experiment(litlogger_mock, tmp_path):
    """Test finalize does nothing if experiment not initialized."""
    logger = LitLogger(name="test", root_dir=tmp_path)

    # Don't initialize the experiment
    logger.finalize("success")

    # finalize should not be called since experiment is None
    litlogger_mock.Experiment.return_value.finalize.assert_not_called()


def test_litlogger_log_file(litlogger_mock, tmp_path):
    """Test log_file method."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    logger.log_file("config.yaml")

    litlogger_mock.File.assert_called_once_with("config.yaml")
    litlogger_mock.Experiment.return_value.__setitem__.assert_any_call("config.yaml", litlogger_mock.File.return_value)


def test_litlogger_get_file(litlogger_mock, tmp_path):
    """Test get_file method."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    result = logger.get_file("config.yaml", verbose=True)

    litlogger_mock.Experiment.return_value.__getitem__.assert_any_call("config.yaml")
    litlogger_mock.Experiment.return_value.series_mocks["config.yaml"].save.assert_called_once_with("config.yaml")
    assert result == litlogger_mock.Experiment.return_value.series_mocks["config.yaml"].save.return_value


def test_litlogger_log_model(litlogger_mock, tmp_path):
    """Test log_model method."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    model = torch.nn.Linear(10, 10)
    logger.log_model(model, staging_dir="/tmp", verbose=True, version="v1", metadata={"epoch": 10})

    litlogger_mock.Model.assert_called_once_with(model, version="v1", metadata={"epoch": 10}, staging_dir="/tmp")
    litlogger_mock.Experiment.return_value.__setitem__.assert_any_call(
        logger._experiment_name, litlogger_mock.Model.return_value
    )


def test_litlogger_log_model_uses_step_as_default_version(litlogger_mock, tmp_path):
    """Test log_model defaults the model version to the current step."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    logger._step = 7
    model = torch.nn.Linear(10, 10)

    logger.log_model(model)

    litlogger_mock.Model.assert_called_once_with(model, version="7", metadata=None, staging_dir=None)


def test_litlogger_log_model_artifact(litlogger_mock, tmp_path):
    """Test log_model_artifact method."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    logger.log_model_artifact("/path/to/model.ckpt", verbose=True, version="v1")

    litlogger_mock.Model.assert_called_once_with("/path/to/model.ckpt", version="v1")
    litlogger_mock.Experiment.return_value.__setitem__.assert_any_call(
        logger._experiment_name, litlogger_mock.Model.return_value
    )


def test_litlogger_log_model_artifact_uses_step_as_default_version(litlogger_mock, tmp_path):
    """Test log_model_artifact defaults the model version to the current step."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    logger._step = 11

    logger.log_model_artifact("/path/to/model.ckpt")

    litlogger_mock.Model.assert_called_once_with("/path/to/model.ckpt", version="11")


def test_litlogger_url_property(litlogger_mock, tmp_path):
    """Test url property."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    url = logger.url

    assert url == "https://lightning.ai/test/experiments/test-experiment"


def test_litlogger_version_property(litlogger_mock, tmp_path):
    """Test version property is set after experiment initialization."""
    logger = LitLogger(name="test", root_dir=tmp_path)

    # Before accessing experiment, version is None
    assert logger.version is None

    # After accessing experiment, version is set
    _ = logger.experiment
    assert logger.version is not None
    assert ":" not in logger.version


def test_litlogger_with_trainer(litlogger_mock, tmp_path):
    """Test LitLogger works with Trainer."""

    class LoggingModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            self.log("train_loss", loss["loss"])
            return loss

    logger = LitLogger(name="test", root_dir=tmp_path)
    model = LoggingModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    trainer.fit(model)

    # Verify metrics were logged
    assert any(series.append.called for series in litlogger_mock.Experiment.return_value.series_mocks.values())


def test_litlogger_metadata_in_init(litlogger_mock, tmp_path):
    """Test metadata is passed to litlogger.Experiment."""
    logger = LitLogger(
        name="test",
        root_dir=tmp_path,
        metadata={"experiment_type": "test", "version": "1.0"},
    )

    _ = logger.experiment

    call_kwargs = litlogger_mock.Experiment.call_args[1]
    assert call_kwargs["name"].startswith("test-")
    assert call_kwargs["name"].endswith(logger.version)
    assert ":" not in logger.version
    assert call_kwargs["metadata"] == {"experiment_type": "test", "version": "1.0"}


def test_litlogger_log_model_disabled(litlogger_mock, tmp_path):
    """Test log_model option defaults to False."""
    logger = LitLogger(name="test", root_dir=tmp_path)
    assert logger._log_model is False


def test_litlogger_log_model_enabled(litlogger_mock, tmp_path):
    """Test log_model option can be enabled."""
    logger = LitLogger(name="test", root_dir=tmp_path, log_model=True)
    assert logger._log_model is True


def test_litlogger_after_save_checkpoint_disabled(litlogger_mock, tmp_path):
    """Test after_save_checkpoint does nothing when log_model=False."""
    logger = LitLogger(name="test", root_dir=tmp_path, log_model=False)
    checkpoint_callback = MagicMock()
    checkpoint_callback.save_top_k = 1

    logger.after_save_checkpoint(checkpoint_callback)

    # Should not set checkpoint callback
    assert logger._checkpoint_callback is None


def test_litlogger_after_save_checkpoint_enabled(litlogger_mock, tmp_path):
    """Test after_save_checkpoint stores callback when log_model=True."""
    logger = LitLogger(name="test", root_dir=tmp_path, log_model=True)
    checkpoint_callback = MagicMock()
    checkpoint_callback.save_top_k = 1

    logger.after_save_checkpoint(checkpoint_callback)

    # Should store checkpoint callback for later
    assert logger._checkpoint_callback is checkpoint_callback


def test_litlogger_save_logs_option(litlogger_mock, tmp_path):
    """Test save_logs option is passed to Experiment."""
    logger = LitLogger(name="test", root_dir=tmp_path, save_logs=True)

    _ = logger.experiment

    call_kwargs = litlogger_mock.Experiment.call_args[1]
    assert call_kwargs["save_logs"] is True
