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


def test_file_logger_automatic_versioning(tmpdir):
    """Verify that automatic versioning works."""
    root_dir = tmpdir.mkdir("exp")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")
    logger = CSVLogger(root_dir=root_dir, name="exp")
    assert logger.version == 2


def test_file_logger_manual_versioning(tmpdir):
    """Verify that manual versioning works."""
    root_dir = tmpdir.mkdir("exp")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")
    root_dir.mkdir("version_2")
    logger = CSVLogger(root_dir=root_dir, name="exp", version=1)
    assert logger.version == 1


def test_file_logger_named_version(tmpdir):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    exp_name = "exp"
    tmpdir.mkdir(exp_name)
    expected_version = "2020-02-05-162402"

    logger = CSVLogger(root_dir=tmpdir, name=exp_name, version=expected_version)
    logger.log_metrics({"a": 1, "b": 2})
    logger.save()
    assert logger.version == expected_version
    assert os.listdir(tmpdir / exp_name) == [expected_version]
    assert os.listdir(tmpdir / exp_name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_file_logger_no_name(tmpdir, name):
    """Verify that None or empty name works."""
    logger = CSVLogger(root_dir=tmpdir, name=name)
    logger.log_metrics({"a": 1})
    logger.save()
    assert os.path.normpath(logger.root_dir) == tmpdir  # use os.path.normpath to handle trailing /
    assert os.listdir(tmpdir / "version_0")


@pytest.mark.parametrize("step_idx", [10, None])
def test_file_logger_log_metrics(tmpdir, step_idx):
    logger = CSVLogger(tmpdir)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)
    logger.save()

    path_csv = os.path.join(logger.log_dir, _ExperimentWriter.NAME_METRICS_FILE)
    with open(path_csv) as fp:
        lines = fp.readlines()
    assert len(lines) == 2
    assert all(n in lines[0] for n in metrics)


def test_file_logger_log_hyperparams(tmpdir):
    logger = CSVLogger(tmpdir)
    with pytest.raises(NotImplementedError):
        logger.log_hyperparams({})


def test_flush_n_steps(tmpdir):
    logger = CSVLogger(tmpdir, flush_logs_every_n_steps=2)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.save = MagicMock()
    logger.log_metrics(metrics, step=0)

    logger.save.assert_not_called()
    logger.log_metrics(metrics, step=1)
    logger.save.assert_called_once()
