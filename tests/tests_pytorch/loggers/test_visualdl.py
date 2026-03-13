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
from argparse import Namespace
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import VisualDLLogger
from lightning.pytorch.loggers.visualdl import _VISUALDL_AVAILABLE
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from tests_pytorch.helpers.runif import RunIf

if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_hparams_reload(tmp_path):
    """Test that hyperparameters are properly saved and reloaded."""

    class CustomModel(BoringModel):
        def __init__(self, b1=0.5, b2=0.999):
            super().__init__()
            self.save_hyperparameters()

    trainer = Trainer(max_steps=1, default_root_dir=tmp_path, logger=VisualDLLogger(tmp_path))
    model = CustomModel()
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model)

    assert trainer.log_dir == trainer.logger.log_dir
    folder_path = trainer.log_dir

    # make sure yaml is there
    hparams_file = os.path.join(folder_path, "hparams.yaml")
    assert os.path.isfile(hparams_file)

    with open(hparams_file) as file:
        yaml_params = yaml.safe_load(file)
        assert yaml_params["b1"] == 0.5
        assert yaml_params["b2"] == 0.999
        assert len(yaml_params.keys()) == 2

    # verify artifacts
    assert len(os.listdir(os.path.join(folder_path, "checkpoints"))) == 1


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    root_dir = tmp_path / "vdl_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_nonumber").mkdir()
    (root_dir / "other").mkdir()

    logger = VisualDLLogger(save_dir=tmp_path, name="vdl_versioning")
    assert logger.version == 2


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path / "vdl_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = VisualDLLogger(save_dir=tmp_path, name="vdl_versioning", version=1)
    assert logger.version == 1


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    name = "vdl_versioning"
    (tmp_path / name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = VisualDLLogger(save_dir=tmp_path, name=name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written

    assert logger.version == expected_version
    assert os.listdir(tmp_path / name) == [expected_version]
    assert os.listdir(tmp_path / name / expected_version)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
@pytest.mark.parametrize("name", ["", None])
def test_visualdl_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = VisualDLLogger(save_dir=tmp_path, name=name)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written
    assert os.path.normpath(logger.root_dir) == str(tmp_path)  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_sub_dir(tmp_path):
    """Test logging with subdirectory."""

    class TestLogger(VisualDLLogger):
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

    with mock.patch.dict(os.environ, {"TEST_ENV_DIR": "some_directory"}):
        # test env var (`$`) handling
        save_dir = "$TEST_ENV_DIR/tmp"
        explicit_save_dir = f"{os.environ['TEST_ENV_DIR']}/tmp"
        logger = TestLogger(save_dir, sub_dir="sub_dir")
        trainer = Trainer(**trainer_args, logger=logger)
        assert trainer.logger.log_dir == os.path.join(explicit_save_dir, "name", "version", "sub_dir")


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
@pytest.mark.parametrize("step_idx", [10, None])
def test_visualdl_log_metrics(tmp_path, step_idx):
    """Test logging metrics."""
    logger = VisualDLLogger(tmp_path)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_hyperparams(tmp_path):
    """Test logging hyperparameters."""
    logger = VisualDLLogger(tmp_path)
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


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_hparams_and_metrics(tmp_path):
    """Test logging hyperparameters with metrics."""
    logger = VisualDLLogger(tmp_path, default_hp_metric=False)
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
@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_omegaconf_hparams_and_metrics(tmp_path):
    """Test logging OmegaConf hyperparameters."""
    logger = VisualDLLogger(tmp_path, default_hp_metric=False)
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


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 32)])
def test_visualdl_log_graph(tmp_path, example_input_array):
    """Test that log graph works (or warns) appropriately."""
    model = BoringModel()
    if example_input_array is not None:
        model.example_input_array = None

    logger = VisualDLLogger(tmp_path, log_graph=True)
    logger.log_graph(model, example_input_array)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_graph_warning_no_example_input_array(tmp_path):
    """Test that log graph throws warning if model.example_input_array is None."""
    model = BoringModel()
    model.example_input_array = None
    logger = VisualDLLogger(tmp_path, log_graph=True)
    with pytest.warns(
        UserWarning,
        match="Could not log computational graph to VisualDL: The `model.example_input_array` .* was not given",
    ):
        logger.log_graph(model)


@mock.patch("lightning.pytorch.loggers.VisualDLLogger.log_metrics")
@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_with_accumulated_gradients(mock_log_metrics, tmp_path):
    """Tests to ensure that visualdl log properly when accumulated_gradients > 1."""

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
    logger_0 = VisualDLLogger(tmp_path, default_hp_metric=False)
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


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_finalize(monkeypatch, tmp_path):
    """Test that the LogWriter closes in finalize."""
    from visualdl import LogWriter

    monkeypatch.setattr(LogWriter, "flush", Mock())
    monkeypatch.setattr(LogWriter, "close", Mock())

    logger = VisualDLLogger(save_dir=tmp_path)
    assert logger._experiment is None
    logger.finalize("any")

    # no log calls, no experiment created -> nothing to flush
    assert not hasattr(logger, "experiment") or logger._experiment is None

    logger = VisualDLLogger(save_dir=tmp_path)
    logger.log_metrics({"flush_me": 11.1})  # trigger creation of an experiment
    logger.finalize("any")

    # finalize flushes to experiment directory
    if logger._experiment is not None:
        logger._experiment.flush.assert_called()
        logger._experiment.close.assert_called()


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_save_hparams_to_yaml_once(tmp_path):
    """Test that hparams.yaml is saved only once in the correct location."""
    model = BoringModel()
    logger = VisualDLLogger(save_dir=tmp_path, default_hp_metric=False)
    trainer = Trainer(max_steps=1, default_root_dir=tmp_path, logger=logger)
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model)

    hparams_file = "hparams.yaml"
    assert os.path.isfile(os.path.join(trainer.log_dir, hparams_file))
    assert not os.path.isfile(os.path.join(tmp_path, hparams_file))


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_with_symlink(tmp_path, monkeypatch):
    """Tests a specific failure case when visualdl logger is used with empty name, symbolic link ``save_dir``, and
    relative paths."""
    monkeypatch.chdir(tmp_path)  # need to use relative paths
    source = os.path.join(".", "lightning_logs")
    dest = os.path.join(".", "sym_lightning_logs")

    os.makedirs(source, exist_ok=True)
    os.symlink(source, dest)

    logger = VisualDLLogger(save_dir=dest, name="")
    _ = logger.version


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_display_name_and_file_name(tmp_path):
    """Test setting display_name and file_name parameters."""
    display_name = "My Experiment"
    file_name = "custom_vdlrecords.log"

    logger = VisualDLLogger(save_dir=tmp_path, name="test", display_name=display_name, file_name=file_name)

    # Force experiment creation
    logger.log_metrics({"test": 1.0})

    assert logger._display_name == display_name
    assert logger._file_name == file_name


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_max_queue_and_flush_secs(tmp_path):
    """Test setting max_queue and flush_secs parameters."""
    max_queue = 20
    flush_secs = 60

    logger = VisualDLLogger(save_dir=tmp_path, name="test", max_queue=max_queue, flush_secs=flush_secs)

    assert logger._max_queue == max_queue
    assert logger._flush_secs == flush_secs
