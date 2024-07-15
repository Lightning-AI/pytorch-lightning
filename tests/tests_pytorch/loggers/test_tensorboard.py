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
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.tensorboard import _TENSORBOARD_AVAILABLE
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE

from tests_pytorch.helpers.runif import RunIf

if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf


def test_tensorboard_hparams_reload(tmp_path):
    class CustomModel(BoringModel):
        def __init__(self, b1=0.5, b2=0.999):
            super().__init__()
            self.save_hyperparameters()

    trainer = Trainer(max_steps=1, default_root_dir=tmp_path, logger=TensorBoardLogger(tmp_path))
    model = CustomModel()
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model)

    assert trainer.log_dir == trainer.logger.log_dir
    folder_path = trainer.log_dir

    # make sure yaml is there
    with open(os.path.join(folder_path, "hparams.yaml")) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        yaml_params = yaml.safe_load(file)
        assert yaml_params["b1"] == 0.5
        assert yaml_params["b2"] == 0.999
        assert len(yaml_params.keys()) == 2

    # verify artifacts
    assert len(os.listdir(os.path.join(folder_path, "checkpoints"))) == 1


def test_tensorboard_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    root_dir = tmp_path / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_nonumber").mkdir()
    (root_dir / "other").mkdir()

    logger = TensorBoardLogger(save_dir=tmp_path, name="tb_versioning")
    assert logger.version == 2


def test_tensorboard_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = TensorBoardLogger(save_dir=tmp_path, name="tb_versioning", version=1)

    assert logger.version == 1


def test_tensorboard_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    name = "tb_versioning"
    (tmp_path / name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = TensorBoardLogger(save_dir=tmp_path, name=name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written

    assert logger.version == expected_version
    assert os.listdir(tmp_path / name) == [expected_version]
    assert os.listdir(tmp_path / name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_tensorboard_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = TensorBoardLogger(save_dir=tmp_path, name=name)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written
    assert os.path.normpath(logger.root_dir) == str(tmp_path)  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


def test_tensorboard_log_sub_dir(tmp_path):
    class TestLogger(TensorBoardLogger):
        # for reproducibility
        @property
        def version(self):
            return "version"

        @property
        def name(self):
            return "name"

    trainer_args = {"default_root_dir": tmp_path, "max_steps": 1}

    # no sub_dir specified
    save_dir = tmp_path / "logs"
    logger = TestLogger(save_dir)
    trainer = Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(save_dir, "name", "version")

    # sub_dir specified
    logger = TestLogger(save_dir, sub_dir="sub_dir")
    trainer = Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(save_dir, "name", "version", "sub_dir")

    # test home dir (`~`) handling
    save_dir = "~/tmp"
    explicit_save_dir = os.path.expanduser(save_dir)
    logger = TestLogger(save_dir, sub_dir="sub_dir")
    trainer = Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(explicit_save_dir, "name", "version", "sub_dir")

    with mock.patch.dict(os.environ, {}):
        # test env var (`$`) handling
        test_env_dir = "some_directory"
        os.environ["TEST_ENV_DIR"] = test_env_dir
        save_dir = "$TEST_ENV_DIR/tmp"
        explicit_save_dir = f"{test_env_dir}/tmp"
        logger = TestLogger(save_dir, sub_dir="sub_dir")
        trainer = Trainer(**trainer_args, logger=logger)
        assert trainer.logger.log_dir == os.path.join(explicit_save_dir, "name", "version", "sub_dir")


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_metrics(tmp_path, step_idx):
    logger = TensorBoardLogger(tmp_path)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)


def test_tensorboard_log_hyperparams(tmp_path):
    logger = TensorBoardLogger(tmp_path)
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


def test_tensorboard_log_hparams_and_metrics(tmp_path):
    logger = TensorBoardLogger(tmp_path, default_hp_metric=False)
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


@RunIf(omegaconf=True)
def test_tensorboard_log_omegaconf_hparams_and_metrics(tmp_path):
    logger = TensorBoardLogger(tmp_path, default_hp_metric=False)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
    }
    hparams = OmegaConf.create(hparams)

    metrics = {"abc": torch.tensor([0.54])}
    logger.log_hyperparams(hparams, metrics)


@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 32)])
def test_tensorboard_log_graph(tmp_path, example_input_array):
    """Test that log graph works with both model.example_input_array and if array is passed externally."""
    model = BoringModel()
    if example_input_array is not None:
        model.example_input_array = None

    logger = TensorBoardLogger(tmp_path, log_graph=True)
    logger.log_graph(model, example_input_array)


@pytest.mark.skipif(not _TENSORBOARD_AVAILABLE, reason=str(_TENSORBOARD_AVAILABLE))
def test_tensorboard_log_graph_warning_no_example_input_array(tmp_path):
    """Test that log graph throws warning if model.example_input_array is None."""
    model = BoringModel()
    model.example_input_array = None
    logger = TensorBoardLogger(tmp_path, log_graph=True)
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


@mock.patch("lightning.pytorch.loggers.TensorBoardLogger.log_metrics")
def test_tensorboard_with_accummulated_gradients(mock_log_metrics, tmp_path):
    """Tests to ensure that tensorboard log properly when accumulated_gradients > 1."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.indexes = []

        def training_step(self, *args):
            self.log("foo", 1, on_step=True, on_epoch=True)
            if not self.trainer.fit_loop._should_accumulate() and self.trainer._logger_connector.should_update_logs:
                self.indexes.append(self.trainer.global_step)
            return super().training_step(*args)

    model = TestModel()
    logger_0 = TensorBoardLogger(tmp_path, default_hp_metric=False)
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=12,
        limit_val_batches=0,
        max_epochs=3,
        accumulate_grad_batches=2,
        logger=[logger_0],
        log_every_n_steps=3,
    )
    trainer.fit(model)

    calls = [m[2] for m in mock_log_metrics.mock_calls]
    count_epochs = [c["step"] for c in calls if "foo_epoch" in c["metrics"]]
    assert count_epochs == [5, 11, 17]

    count_steps = [c["step"] for c in calls if "foo_step" in c["metrics"]]
    assert count_steps == model.indexes


def test_tensorboard_finalize(monkeypatch, tmp_path):
    """Test that the SummaryWriter closes in finalize."""
    if _TENSORBOARD_AVAILABLE:
        import torch.utils.tensorboard as tb
    else:
        import tensorboardX as tb

    monkeypatch.setattr(tb, "SummaryWriter", Mock())
    logger = TensorBoardLogger(save_dir=tmp_path)
    assert logger._experiment is None
    logger.finalize("any")

    # no log calls, no experiment created -> nothing to flush
    logger.experiment.assert_not_called()

    logger = TensorBoardLogger(save_dir=tmp_path)
    logger.log_metrics({"flush_me": 11.1})  # trigger creation of an experiment
    logger.finalize("any")

    # finalize flushes to experiment directory
    logger.experiment.flush.assert_called()
    logger.experiment.close.assert_called()


def test_tensorboard_save_hparams_to_yaml_once(tmp_path):
    model = BoringModel()
    logger = TensorBoardLogger(save_dir=tmp_path, default_hp_metric=False)
    trainer = Trainer(max_steps=1, default_root_dir=tmp_path, logger=logger)
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model)

    hparams_file = "hparams.yaml"
    assert os.path.isfile(os.path.join(trainer.log_dir, hparams_file))
    assert not os.path.isfile(os.path.join(tmp_path, hparams_file))


@mock.patch("lightning.pytorch.loggers.tensorboard.log")
def test_tensorboard_with_symlink(log, tmp_path, monkeypatch):
    """Tests a specific failure case when tensorboard logger is used with empty name, symbolic link ``save_dir``, and
    relative paths."""
    monkeypatch.chdir(tmp_path)  # need to use relative paths
    source = os.path.join(".", "lightning_logs")
    dest = os.path.join(".", "sym_lightning_logs")

    os.makedirs(source, exist_ok=True)
    os.symlink(source, dest)

    logger = TensorBoardLogger(save_dir=dest, name="")
    _ = logger.version

    log.warning.assert_not_called()


def test_tensorboard_missing_folder_warning(tmp_path, caplog):
    """Verify that the logger throws a warning for invalid directory."""
    name = "fake_dir"
    logger = TensorBoardLogger(save_dir=tmp_path, name=name)

    with caplog.at_level(logging.WARNING):
        assert logger.version == 0

    assert "Missing logger folder:" in caplog.text
