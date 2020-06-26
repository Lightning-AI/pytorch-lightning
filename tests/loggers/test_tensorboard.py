import os
from argparse import Namespace

import pytest
import torch
import yaml
from packaging import version
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tests.base import EvalModelTemplate


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.5.0'),
                    reason='Minimal PT version is set to 1.5')
def test_tensorboard_hparams_reload(tmpdir):
    model = EvalModelTemplate()

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)

    folder_path = trainer.logger.log_dir

    # make sure yaml is there
    with open(os.path.join(folder_path, 'hparams.yaml')) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        yaml_params = yaml.load(file, Loader=yaml.FullLoader)
        assert yaml_params['b1'] == 0.5
        assert len(yaml_params.keys()) == 10

    # verify artifacts
    assert len(os.listdir(os.path.join(folder_path, 'checkpoints'))) == 1

    # verify tb logs
    event_acc = EventAccumulator(folder_path)
    event_acc.Reload()

    hparams_data = b'\x12\x84\x01"\x0b\n\tdrop_prob"\x0c\n\nbatch_size"\r\n\x0bin_features"' \
                   b'\x0f\n\rlearning_rate"\x10\n\x0eoptimizer_name"\x0b\n\tdata_root"\x0e\n' \
                   b'\x0cout_features"\x0c\n\nhidden_dim"\x04\n\x02b1"\x04\n\x02b2'

    assert event_acc.summary_metadata['_hparams_/experiment'].plugin_data.plugin_name == 'hparams'
    assert event_acc.summary_metadata['_hparams_/experiment'].plugin_data.content == hparams_data


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
