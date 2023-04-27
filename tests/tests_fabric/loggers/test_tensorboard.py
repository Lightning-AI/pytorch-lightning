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
import logging
import os
from argparse import Namespace
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE
from tests_fabric.test_fabric import BoringModel


def test_tensorboard_automatic_versioning(tmpdir):
    """Verify that automatic versioning works."""
    root_dir = tmpdir / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()

    logger = TensorBoardLogger(root_dir=tmpdir, name="tb_versioning")
    assert logger.version == 2


def test_tensorboard_manual_versioning(tmpdir):
    """Verify that manual versioning works."""
    root_dir = tmpdir / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = TensorBoardLogger(root_dir=tmpdir, name="tb_versioning", version=1)
    assert logger.version == 1


def test_tensorboard_named_version(tmpdir):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    name = "tb_versioning"
    (tmpdir / name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = TensorBoardLogger(root_dir=tmpdir, name=name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written

    assert logger.version == expected_version
    assert os.listdir(tmpdir / name) == [expected_version]
    assert os.listdir(tmpdir / name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_tensorboard_no_name(tmpdir, name):
    """Verify that None or empty name works."""
    logger = TensorBoardLogger(root_dir=tmpdir, name=name)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written
    assert os.path.normpath(logger.root_dir) == tmpdir  # use os.path.normpath to handle trailing /
    assert os.listdir(tmpdir / "version_0")


def test_tensorboard_log_sub_dir(tmpdir):
    # no sub_dir specified
    root_dir = tmpdir / "logs"
    logger = TensorBoardLogger(root_dir, name="name", version="version")
    assert logger.log_dir == os.path.join(root_dir, "name", "version")

    # sub_dir specified
    logger = TensorBoardLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.log_dir == os.path.join(root_dir, "name", "version", "sub_dir")


def test_tensorboard_expand_home():
    """Test that the home dir (`~`) gets expanded properly."""
    root_dir = "~/tmp"
    explicit_root_dir = os.path.expanduser(root_dir)
    logger = TensorBoardLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.root_dir == root_dir
    assert logger.log_dir == os.path.join(explicit_root_dir, "name", "version", "sub_dir")


@mock.patch.dict(os.environ, {"TEST_ENV_DIR": "some_directory"})
def test_tensorboard_expand_env_vars():
    """Test that the env vars in path names (`$`) get handled properly."""
    test_env_dir = os.environ["TEST_ENV_DIR"]
    root_dir = "$TEST_ENV_DIR/tmp"
    explicit_root_dir = f"{test_env_dir}/tmp"
    logger = TensorBoardLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.log_dir == os.path.join(explicit_root_dir, "name", "version", "sub_dir")


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_metrics(tmpdir, step_idx):
    logger = TensorBoardLogger(tmpdir)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
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
        "tensor": torch.empty(2, 2, 2),
        "array": np.empty([2, 2, 2]),
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
        "tensor": torch.empty(2, 2, 2),
        "array": np.empty([2, 2, 2]),
    }
    metrics = {"abc": torch.tensor([0.54])}
    logger.log_hyperparams(hparams, metrics)


@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 32)])
def test_tensorboard_log_graph(tmpdir, example_input_array):
    """test that log graph works with both model.example_input_array and if array is passed externally."""
    # TODO(fabric): Test both nn.Module and LightningModule
    # TODO(fabric): Assert _apply_batch_transfer_handler is calling the batch transfer hooks
    model = BoringModel()
    if example_input_array is not None:
        model.example_input_array = None

    logger = TensorBoardLogger(tmpdir, log_graph=True)
    logger.log_graph(model, example_input_array)


@pytest.mark.skipif(not _TENSORBOARD_AVAILABLE, reason=str(_TENSORBOARD_AVAILABLE))
def test_tensorboard_log_graph_warning_no_example_input_array(tmpdir):
    """test that log graph throws warning if model.example_input_array is None."""
    model = BoringModel()
    model.example_input_array = None
    logger = TensorBoardLogger(tmpdir, log_graph=True)
    with pytest.warns(
        UserWarning,
        match="Could not log computational graph to TensorBoard: The `model.example_input_array` .* was not given",
    ):
        logger.log_graph(model)

    model.example_input_array = {"x": 1, "y": 2}
    with pytest.warns(
        UserWarning, match="Could not log computational graph to TensorBoard: .* can't be traced by TensorBoard"
    ):
        logger.log_graph(model)


def test_tensorboard_finalize(monkeypatch, tmpdir):
    """Test that the SummaryWriter closes in finalize."""
    if _TENSORBOARD_AVAILABLE:
        import torch.utils.tensorboard as tb
    else:
        import tensorboardX as tb

    monkeypatch.setattr(tb, "SummaryWriter", Mock())
    logger = TensorBoardLogger(root_dir=tmpdir)
    assert logger._experiment is None
    logger.finalize("any")

    # no log calls, no experiment created -> nothing to flush
    logger.experiment.assert_not_called()

    logger = TensorBoardLogger(root_dir=tmpdir)
    logger.log_metrics({"flush_me": 11.1})  # trigger creation of an experiment
    logger.finalize("any")

    # finalize flushes to experiment directory
    logger.experiment.flush.assert_called()
    logger.experiment.close.assert_called()


@mock.patch("lightning.fabric.loggers.tensorboard.log")
def test_tensorboard_with_symlink(log, tmpdir):
    """Tests a specific failure case when tensorboard logger is used with empty name, symbolic link ``save_dir``,
    and relative paths."""
    os.chdir(tmpdir)  # need to use relative paths
    source = os.path.join(".", "lightning_logs")
    dest = os.path.join(".", "sym_lightning_logs")

    os.makedirs(source, exist_ok=True)
    os.symlink(source, dest)

    logger = TensorBoardLogger(root_dir=dest, name="")
    _ = logger.version

    log.warning.assert_not_called()


def test_tensorboard_missing_folder_warning(tmpdir, caplog):
    """Verify that the logger throws a warning for invalid directory."""
    name = "fake_dir"
    logger = TensorBoardLogger(root_dir=tmpdir, name=name)

    with caplog.at_level(logging.WARNING):
        assert logger.version == 0

    assert "Missing logger folder:" in caplog.text
