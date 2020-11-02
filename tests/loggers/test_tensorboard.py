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
import os
from argparse import Namespace
from distutils.version import LooseVersion

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from tests.base import EvalModelTemplate, BoringModel


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.5.0"),
    reason="Minimal PT version is set to 1.5",
)
def test_tensorboard_hparams_reload(tmpdir):
    model = EvalModelTemplate()

    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)

    folder_path = trainer.logger.log_dir

    # make sure yaml is there
    with open(os.path.join(folder_path, "hparams.yaml")) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        yaml_params = yaml.safe_load(file)
        assert yaml_params["b1"] == 0.5
        assert len(yaml_params.keys()) == 10

    # verify artifacts
    assert len(os.listdir(os.path.join(folder_path, "checkpoints"))) == 1

    # verify tb logs
    event_acc = EventAccumulator(folder_path)
    event_acc.Reload()

    data_pt_1_5 = b'\x12\x93\x01"\x0b\n\tdrop_prob"\x0c\n\nbatch_size"\r\n\x0bin_features"\x0f\n\rlearning_rate"' \
                  b'\x10\n\x0eoptimizer_name"\x0b\n\tdata_root"\x0e\n\x0cout_features"\x0c\n\nhidden_dim"' \
                  b'\x04\n\x02b1"\x04\n\x02b2*\r\n\x0b\x12\thp_metric'
    data_pt_1_6 = b'\x12\xa7\x01"\r\n\tdrop_prob \x03"\x0e\n\nbatch_size \x03"\x0f\n\x0bin_features \x03"' \
                  b'\x11\n\rlearning_rate \x03"\x12\n\x0eoptimizer_name \x01"\r\n\tdata_root \x01"' \
                  b'\x10\n\x0cout_features \x03"\x0e\n\nhidden_dim \x03"\x06\n\x02b1 \x03"' \
                  b'\x06\n\x02b2 \x03*\r\n\x0b\x12\thp_metric'

    hparams_data = data_pt_1_6 if LooseVersion(torch.__version__) >= LooseVersion("1.6.0") else data_pt_1_5

    assert event_acc.summary_metadata['_hparams_/experiment'].plugin_data.plugin_name == 'hparams'
    assert event_acc.summary_metadata['_hparams_/experiment'].plugin_data.content == hparams_data


def test_tensorboard_automatic_versioning(tmpdir):
    """Verify that automatic versioning works"""

    root_dir = tmpdir / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning")
    assert logger.version == 2


def test_tensorboard_manual_versioning(tmpdir):
    """Verify that manual versioning works"""

    root_dir = tmpdir / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning", version=1)

    assert logger.version == 1


def test_tensorboard_named_version(tmpdir):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402' """

    name = "tb_versioning"
    (tmpdir / name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = TensorBoardLogger(save_dir=tmpdir, name=name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2})  # Force data to be written

    assert logger.version == expected_version
    assert os.listdir(tmpdir / name) == [expected_version]
    assert os.listdir(tmpdir / name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_tensorboard_no_name(tmpdir, name):
    """Verify that None or empty name works"""
    logger = TensorBoardLogger(save_dir=tmpdir, name=name)
    logger.log_hyperparams({"a": 1, "b": 2})  # Force data to be written
    assert logger.root_dir == tmpdir
    assert os.listdir(tmpdir / "version_0")


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_metrics(tmpdir, step_idx):
    logger = TensorBoardLogger(tmpdir)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": torch.tensor(0.1),
        "IntTensor": torch.tensor(1),
    }
    logger.log_metrics(metrics, step_idx)


def test_tensorboard_log_hyperparams(tmpdir):
    logger = TensorBoardLogger(tmpdir)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar="buzz")),
        "layer": torch.nn.BatchNorm1d,
    }
    logger.log_hyperparams(hparams)


def test_tensorboard_log_hparams_and_metrics(tmpdir):
    logger = TensorBoardLogger(tmpdir, default_hp_metric=False)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar="buzz")),
        "layer": torch.nn.BatchNorm1d,
    }
    metrics = {"abc": torch.tensor([0.54])}
    logger.log_hyperparams(hparams, metrics)


def test_tensorboard_log_omegaconf_hparams_and_metrics(tmpdir):
    logger = TensorBoardLogger(tmpdir, default_hp_metric=False)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        # "namespace": Namespace(foo=Namespace(bar="buzz")),
        # "layer": torch.nn.BatchNorm1d,
    }
    hparams = OmegaConf.create(hparams)

    metrics = {"abc": torch.tensor([0.54])}
    logger.log_hyperparams(hparams, metrics)


@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 28 * 28)])
def test_tensorboard_log_graph(tmpdir, example_input_array):
    """ test that log graph works with both model.example_input_array and
        if array is passed externaly
    """
    model = EvalModelTemplate()
    if example_input_array is not None:
        model.example_input_array = None
    logger = TensorBoardLogger(tmpdir, log_graph=True)
    logger.log_graph(model, example_input_array)


def test_tensorboard_log_graph_warning_no_example_input_array(tmpdir):
    """ test that log graph throws warning if model.example_input_array is None """
    model = EvalModelTemplate()
    model.example_input_array = None
    logger = TensorBoardLogger(tmpdir, log_graph=True)
    with pytest.warns(
        UserWarning,
        match='Could not log computational graph since the `model.example_input_array`'
            ' attribute is not set or `input_array` was not given'
    ):
        logger.log_graph(model)
