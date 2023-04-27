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

import fsspec
import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.core.saving import load_hparams_from_yaml
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.csv_logs import ExperimentWriter
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


def test_file_logger_automatic_versioning(tmpdir):
    """Verify that automatic versioning works."""
    root_dir = tmpdir.mkdir("exp")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")

    logger = CSVLogger(save_dir=tmpdir, name="exp")

    assert logger.version == 2


def test_file_logger_manual_versioning(tmpdir):
    """Verify that manual versioning works."""
    root_dir = tmpdir.mkdir("exp")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")
    root_dir.mkdir("version_2")

    logger = CSVLogger(save_dir=tmpdir, name="exp", version=1)

    assert logger.version == 1


def test_file_logger_named_version(tmpdir):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    exp_name = "exp"
    tmpdir.mkdir(exp_name)
    expected_version = "2020-02-05-162402"

    logger = CSVLogger(save_dir=tmpdir, name=exp_name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2})
    logger.save()
    assert logger.version == expected_version
    assert os.listdir(tmpdir / exp_name) == [expected_version]
    assert os.listdir(tmpdir / exp_name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_file_logger_no_name(tmpdir, name):
    """Verify that None or empty name works."""
    logger = CSVLogger(save_dir=tmpdir, name=name)
    logger.save()
    assert os.path.normpath(logger.root_dir) == tmpdir  # use os.path.normpath to handle trailing /
    assert os.listdir(tmpdir / "version_0")


@pytest.mark.parametrize("step_idx", [10, None])
def test_file_logger_log_metrics(tmpdir, step_idx):
    logger = CSVLogger(tmpdir)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)
    logger.save()

    path_csv = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    with open(path_csv) as fp:
        lines = fp.readlines()
    assert len(lines) == 2
    assert all(n in lines[0] for n in metrics)


def test_file_logger_log_hyperparams(tmpdir):
    logger = CSVLogger(tmpdir)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "layer": torch.nn.BatchNorm1d,
    }
    logger.log_hyperparams(hparams)
    logger.save()

    path_yaml = os.path.join(logger.log_dir, ExperimentWriter.NAME_HPARAMS_FILE)
    params = load_hparams_from_yaml(path_yaml)
    assert all(n in params for n in hparams)


@RunIf(sklearn=True)
def test_fit_csv_logger(tmpdir):
    dm = ClassifDataModule()
    model = ClassificationModel()
    logger = CSVLogger(save_dir=tmpdir)
    trainer = Trainer(default_root_dir=tmpdir, max_steps=10, logger=logger, log_every_n_steps=1)
    trainer.fit(model, datamodule=dm)
    metrics_file = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    assert os.path.isfile(metrics_file)


def test_csv_logger_remotefs():
    logger = CSVLogger(save_dir="memory://test_fit_csv_logger_remotefs")
    fs, _ = fsspec.core.url_to_fs("memory://test_fit_csv_logger_remotefs")
    exp = logger.experiment
    exp.log_metrics({"loss": 0.1})
    exp.save()
    metrics_file = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    assert fs.isfile(metrics_file)


def test_flush_n_steps(tmpdir):
    logger = CSVLogger(tmpdir, flush_logs_every_n_steps=2)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.save = MagicMock()
    logger.log_metrics(metrics, step=0)

    logger.save.assert_not_called()
    logger.log_metrics(metrics, step=1)
    logger.save.assert_called_once()
