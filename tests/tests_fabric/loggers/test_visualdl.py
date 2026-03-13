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

from lightning.fabric.loggers import VisualDLLogger
from lightning.fabric.loggers.visualdl import _VISUALDL_AVAILABLE
from lightning.fabric.wrappers import _FabricModule
from tests_fabric.test_fabric import BoringModel


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    root_dir = tmp_path / "vdl_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_nonumber").mkdir()
    (root_dir / "other").mkdir()

    logger = VisualDLLogger(root_dir=tmp_path, name="vdl_versioning")
    assert logger.version == 2


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path / "vdl_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = VisualDLLogger(root_dir=tmp_path, name="vdl_versioning", version=1)
    assert logger.version == 1


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    name = "vdl_versioning"
    (tmp_path / name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = VisualDLLogger(root_dir=tmp_path, name=name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written

    assert logger.version == expected_version
    assert os.listdir(tmp_path / name) == [expected_version]
    assert os.listdir(tmp_path / name / expected_version)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
@pytest.mark.parametrize("name", ["", None])
def test_visualdl_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = VisualDLLogger(root_dir=tmp_path, name=name)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written
    assert os.path.normpath(logger.root_dir) == str(tmp_path)  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_sub_dir(tmp_path):
    # no sub_dir specified
    root_dir = tmp_path / "logs"
    logger = VisualDLLogger(root_dir, name="name", version="version")
    assert logger.log_dir == os.path.join(root_dir, "name", "version")

    # sub_dir specified
    logger = VisualDLLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.log_dir == os.path.join(root_dir, "name", "version", "sub_dir")


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_expand_home():
    """Test that the home dir (`~`) gets expanded properly."""
    root_dir = "~/tmp"
    explicit_root_dir = os.path.expanduser(root_dir)
    logger = VisualDLLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.root_dir == root_dir
    assert logger.log_dir == os.path.join(explicit_root_dir, "name", "version", "sub_dir")


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
@mock.patch.dict(os.environ, {"TEST_ENV_DIR": "some_directory"})
def test_visualdl_expand_env_vars():
    """Test that the env vars in path names (`$`) get handled properly."""
    test_env_dir = os.environ["TEST_ENV_DIR"]
    root_dir = "$TEST_ENV_DIR/tmp"
    explicit_root_dir = f"{test_env_dir}/tmp"
    logger = VisualDLLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.log_dir == os.path.join(explicit_root_dir, "name", "version", "sub_dir")


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
@pytest.mark.parametrize("step_idx", [10, None])
def test_visualdl_log_metrics(tmp_path, step_idx):
    logger = VisualDLLogger(tmp_path)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_hyperparams(tmp_path):
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


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 32)])
def test_visualdl_log_graph_plain_module(tmp_path, example_input_array):
    model = BoringModel()
    logger = VisualDLLogger(tmp_path)
    logger._experiment = Mock()

    logger.log_graph(model, example_input_array)
    # VisualDL doesn't support add_graph, so it should warn but not crash
    # We just verify the method doesn't raise any exceptions

    logger._experiment.reset_mock()

    wrapped = _FabricModule(model, strategy=Mock())
    logger.log_graph(wrapped, example_input_array)
    # VisualDL doesn't support add_graph, so it should warn but not crash


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 32)])
def test_visualdl_log_graph_with_batch_transfer_hooks(tmp_path, example_input_array):
    model = pytest.importorskip("lightning.pytorch.demos.boring_classes").BoringModel()
    logger = VisualDLLogger(tmp_path)
    logger._experiment = Mock()

    with (
        mock.patch.object(model, "_on_before_batch_transfer", return_value=example_input_array) as before_mock,
        mock.patch.object(model, "_apply_batch_transfer_handler", return_value=example_input_array) as transfer_mock,
    ):
        logger.log_graph(model, example_input_array)
        logger._experiment.reset_mock()

        wrapped = _FabricModule(model, strategy=Mock())
        logger.log_graph(wrapped, example_input_array)

        # VisualDL doesn't support add_graph, but the batch transfer hooks should still be called
        if example_input_array is not None:
            assert before_mock.call_count == 2
            assert transfer_mock.call_count == 2
        else:
            before_mock.assert_not_called()
            transfer_mock.assert_not_called()


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


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_finalize(monkeypatch, tmp_path):
    """Test that the LogWriter closes in finalize."""
    from visualdl import LogWriter

    monkeypatch.setattr(LogWriter, "flush", Mock())
    monkeypatch.setattr(LogWriter, "close", Mock())

    logger = VisualDLLogger(root_dir=tmp_path)
    assert logger._experiment is None
    logger.finalize("any")

    # no log calls, no experiment created -> nothing to flush
    assert not hasattr(logger, "experiment") or logger._experiment is None

    logger = VisualDLLogger(root_dir=tmp_path)
    logger.log_metrics({"flush_me": 11.1})  # trigger creation of an experiment
    logger.finalize("any")

    # finalize flushes and closes the experiment
    if logger._experiment is not None:
        logger._experiment.flush.assert_called()
        logger._experiment.close.assert_called()


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_with_symlink(tmp_path, monkeypatch):
    """Tests a specific failure case when visualdl logger is used with empty name, symbolic link ``save_dir``, and
    relative paths."""
    monkeypatch.chdir(tmp_path)  # need to use relative paths
    source = os.path.join(".", "lightning_logs")
    dest = os.path.join(".", "sym_lightning_logs")

    os.makedirs(source, exist_ok=True)
    os.symlink(source, dest)

    logger = VisualDLLogger(root_dir=dest, name="")
    _ = logger.version


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_image(tmp_path):
    """Test logging an image."""
    logger = VisualDLLogger(tmp_path)

    # Test with numpy array
    img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    logger.log_image(tag="test_image", image=img_array, step=0)

    # Test with tensor
    img_tensor = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
    logger.log_image(tag="test_tensor", image=img_tensor, step=1)

    # Test with file path (create a dummy image file)
    from PIL import Image

    img_path = tmp_path / "test.png"
    Image.fromarray(img_array).save(img_path)
    logger.log_image(tag="test_path", image=str(img_path), step=2)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_histogram(tmp_path):
    """Test logging a histogram."""
    logger = VisualDLLogger(tmp_path)

    values = np.random.randn(1000)
    logger.log_histogram(tag="test_histogram", values=values, step=0)

    # Test with tensor
    values_tensor = torch.randn(1000)
    logger.log_histogram(tag="test_histogram_tensor", values=values_tensor, step=1)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_text(tmp_path):
    """Test logging text."""
    logger = VisualDLLogger(tmp_path)

    logger.log_text(tag="test_text", text="Hello, VisualDL!", step=0)
    logger.log_text(tag="test_text", text="Another message", step=1)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_embeddings(tmp_path):
    """Test logging embeddings."""
    logger = VisualDLLogger(tmp_path)

    # Create sample embeddings
    mat = np.random.randn(10, 5)  # 10 points, 5 dimensions
    metadata = [f"point_{i}" for i in range(10)]

    logger.log_embeddings(tag="test_embeddings", mat=mat, metadata=metadata)

    # Test with 2D metadata
    mat_2d = np.random.randn(5, 3)
    metadata_2d = [[f"label_a_{i}", f"label_b_{i}"] for i in range(5)]
    metadata_header = ["label_a", "label_b"]

    logger.log_embeddings(tag="test_embeddings_2d", mat=mat_2d, metadata=metadata_2d, metadata_header=metadata_header)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_audio(tmp_path):
    """Test logging audio."""
    logger = VisualDLLogger(tmp_path)

    # Create a simple sine wave
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_array = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    logger.log_audio(tag="test_audio", audio=audio_array, step=0, sample_rate=sample_rate)

    # Test with tensor
    audio_tensor = torch.from_numpy(audio_array)
    logger.log_audio(tag="test_audio_tensor", audio=audio_tensor, step=1, sample_rate=sample_rate)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_pr_curve(tmp_path):
    """Test logging PR curve."""
    logger = VisualDLLogger(tmp_path)

    labels = np.random.randint(0, 2, 100)
    predictions = np.random.rand(100)

    logger.log_pr_curve(tag="test_pr_curve", labels=labels, predictions=predictions, step=0)

    # Test with tensor
    labels_tensor = torch.from_numpy(labels)
    predictions_tensor = torch.from_numpy(predictions)
    logger.log_pr_curve(tag="test_pr_curve_tensor", labels=labels_tensor, predictions=predictions_tensor, step=1)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_log_roc_curve(tmp_path):
    """Test logging ROC curve."""
    logger = VisualDLLogger(tmp_path)

    labels = np.random.randint(0, 2, 100)
    predictions = np.random.rand(100)

    logger.log_roc_curve(tag="test_roc_curve", labels=labels, predictions=predictions, step=0)

    # Test with tensor
    labels_tensor = torch.from_numpy(labels)
    predictions_tensor = torch.from_numpy(predictions)
    logger.log_roc_curve(tag="test_roc_curve_tensor", labels=labels_tensor, predictions=predictions_tensor, step=1)


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_display_name_and_file_name(tmp_path):
    """Test setting display_name and file_name parameters."""
    display_name = "My Experiment"
    file_name = "custom_vdlrecords.log"

    logger = VisualDLLogger(root_dir=tmp_path, name="test", display_name=display_name, file_name=file_name)

    # Force experiment creation
    logger.log_metrics({"test": 1.0})

    # These parameters are passed to LogWriter but we can't easily verify them
    # Just ensure no errors occur
    assert logger._display_name == display_name
    assert logger._file_name == file_name


@pytest.mark.skipif(not _VISUALDL_AVAILABLE, reason="visualdl is required")
def test_visualdl_max_queue_and_flush_secs(tmp_path):
    """Test setting max_queue and flush_secs parameters."""
    max_queue = 20
    flush_secs = 60

    logger = VisualDLLogger(root_dir=tmp_path, name="test", max_queue=max_queue, flush_secs=flush_secs)

    assert logger._max_queue == max_queue
    assert logger._flush_secs == flush_secs
