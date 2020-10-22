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
from argparse import Namespace

import pytest
import torch
import os

from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.csv_logs import ExperimentWriter


def test_file_logger_automatic_versioning(tmpdir):
    """Verify that automatic versioning works"""

    root_dir = tmpdir.mkdir("exp")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")

    logger = CSVLogger(save_dir=tmpdir, name="exp")

    assert logger.version == 2


def test_file_logger_manual_versioning(tmpdir):
    """Verify that manual versioning works"""

    root_dir = tmpdir.mkdir("exp")
    root_dir.mkdir("version_0")
    root_dir.mkdir("version_1")
    root_dir.mkdir("version_2")

    logger = CSVLogger(save_dir=tmpdir, name="exp", version=1)

    assert logger.version == 1


def test_file_logger_named_version(tmpdir):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402' """

    exp_name = "exp"
    tmpdir.mkdir(exp_name)
    expected_version = "2020-02-05-162402"

    logger = CSVLogger(save_dir=tmpdir, name=exp_name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2})
    logger.save()
    assert logger.version == expected_version
    assert os.listdir(tmpdir / exp_name) == [expected_version]
    assert os.listdir(tmpdir / exp_name / expected_version)


@pytest.mark.parametrize("name", ['', None])
def test_file_logger_no_name(tmpdir, name):
    """Verify that None or empty name works"""
    logger = CSVLogger(save_dir=tmpdir, name=name)
    logger.save()
    assert logger.root_dir == tmpdir
    assert os.listdir(tmpdir / 'version_0')


@pytest.mark.parametrize("step_idx", [10, None])
def test_file_logger_log_metrics(tmpdir, step_idx):
    logger = CSVLogger(tmpdir)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": torch.tensor(0.1),
        "IntTensor": torch.tensor(1)
    }
    logger.log_metrics(metrics, step_idx)
    logger.save()

    path_csv = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    with open(path_csv, 'r') as fp:
        lines = fp.readlines()
    assert len(lines) == 2
    assert all([n in lines[0] for n in metrics])


def test_file_logger_log_hyperparams(tmpdir):
    logger = CSVLogger(tmpdir)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {'a': {'b': 'c'}},
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar='buzz')),
        "layer": torch.nn.BatchNorm1d
    }
    logger.log_hyperparams(hparams)
    logger.save()

    path_yaml = os.path.join(logger.log_dir, ExperimentWriter.NAME_HPARAMS_FILE)
    params = load_hparams_from_yaml(path_yaml)
    assert all([n in params for n in hparams])
