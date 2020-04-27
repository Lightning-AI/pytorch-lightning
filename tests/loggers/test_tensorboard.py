from argparse import Namespace

import pytest
import torch

from pytorch_lightning.loggers import TensorBoardLogger


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
    # Could also test existence of the directory but this fails
    # in the "minimum requirements" test setup


@pytest.mark.parametrize("name", ['', None])
def test_tensorboard_no_name(tmpdir, name):
    """Verify that None or empty name works"""
    logger = TensorBoardLogger(save_dir=tmpdir, name=name)
    assert logger.root_dir == tmpdir


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
        "bool": True,
        "dict": {'a': {'b': 'c'}},
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar='buzz')),
        "layer": torch.nn.BatchNorm1d
    }
    logger.log_hyperparams(hparams)


def test_tensorboard_log_hparams_and_metrics(tmpdir):
    logger = TensorBoardLogger(tmpdir)
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
    metrics = {'abc': torch.tensor([0.54])}
    logger.log_hyperparams(hparams, metrics)
