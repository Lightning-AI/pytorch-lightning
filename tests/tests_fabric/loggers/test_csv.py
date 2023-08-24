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
from unittest.mock import MagicMock

import pytest
import torch

from lightning.fabric.loggers import CSVLogger
from lightning.fabric.loggers.csv_logs import _ExperimentWriter


def test_file_logger_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    (tmp_path / "exp" / "version_0").mkdir(parents=True)
    (tmp_path / "exp" / "version_1").mkdir()
    logger = CSVLogger(root_dir=tmp_path, name="exp")
    assert logger.version == 2


def test_file_logger_automatic_versioning_relative_root_dir(tmp_path, monkeypatch):
    """Verify that automatic versioning works, when root_dir is given a relative path."""
    (tmp_path / "exp" / "logs" / "version_0").mkdir(parents=True)
    (tmp_path / "exp" / "logs" / "version_1").mkdir()
    monkeypatch.chdir(tmp_path)
    logger = CSVLogger(root_dir="exp", name="logs")
    assert logger.version == 2


def test_file_logger_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path.mkdir("exp")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")
    root_dir.mkdir("version_2")
    logger = CSVLogger(root_dir=root_dir, name="exp", version=1)
    assert logger.version == 1


def test_file_logger_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    exp_name = "exp"
    tmp_path.mkdir(exp_name)
    expected_version = "2020-02-05-162402"

    logger = CSVLogger(root_dir=tmp_path, name=exp_name, version=expected_version)
    logger.log_metrics({"a": 1, "b": 2})
    logger.save()
    assert logger.version == expected_version
    assert os.listdir(tmp_path / exp_name) == [expected_version]
    assert os.listdir(tmp_path / exp_name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_file_logger_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = CSVLogger(root_dir=tmp_path, name=name)
    logger.log_metrics({"a": 1})
    logger.save()
    assert os.path.normpath(logger._root_dir) == tmp_path  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


@pytest.mark.parametrize("step_idx", [10, None])
def test_file_logger_log_metrics(tmp_path, step_idx):
    logger = CSVLogger(tmp_path)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)
    logger.save()

    path_csv = os.path.join(logger.log_dir, _ExperimentWriter.NAME_METRICS_FILE)
    with open(path_csv) as fp:
        lines = fp.readlines()
    assert len(lines) == 2
    assert all(n in lines[0] for n in metrics)


def test_file_logger_log_hyperparams(tmp_path):
    logger = CSVLogger(tmp_path)
    with pytest.raises(NotImplementedError):
        logger.log_hyperparams({})


def test_flush_n_steps(tmp_path):
    logger = CSVLogger(tmp_path, flush_logs_every_n_steps=2)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.save = MagicMock()
    logger.log_metrics(metrics, step=0)

    logger.save.assert_not_called()
    logger.log_metrics(metrics, step=1)
    logger.save.assert_called_once()


def test_automatic_step_tracking(tmp_path):
    """Test that the logger keeps track of the step value if it isn't passed in explicitly."""
    logger = CSVLogger(tmp_path, flush_logs_every_n_steps=3)
    logger.save = MagicMock()
    metrics = {"test": 0.1}
    logger.log_metrics(metrics, step=None)
    logger.save.assert_not_called()
    assert logger.experiment.metrics[0]["step"] == 0
    logger.log_metrics(metrics, step=None)
    logger.save.assert_not_called()
    assert logger.experiment.metrics[1]["step"] == 1
    logger.log_metrics(metrics, step=None)
    logger.save.assert_called_once()
    assert logger.experiment.metrics[2]["step"] == 2
