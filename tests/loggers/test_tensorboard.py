import pickle
from argparse import Namespace

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tests.base import LightningTestModel


def test_tensorboard_logger(tmpdir):
    """Verify that basic functionality of Tensorboard logger works."""

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    logger = TensorBoardLogger(save_dir=tmpdir, name="tensorboard_logger_test")

    trainer_options = dict(max_epochs=1, train_percent_check=0.01, logger=logger)

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print("result finished")
    assert result == 1, "Training failed"


def test_tensorboard_pickle(tmpdir):
    """Verify that pickling trainer with Tensorboard logger works."""

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
    # Could also test existence of the directory but this fails
    # in the "minimum requirements" test setup


def test_tensorboard_no_name(tmpdir):
    """Verify that None or empty name works"""

    logger = TensorBoardLogger(save_dir=tmpdir, name="")
    assert logger.root_dir == tmpdir

    logger = TensorBoardLogger(save_dir=tmpdir, name=None)
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
