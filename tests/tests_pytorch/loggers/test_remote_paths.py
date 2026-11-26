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
import ntpath
import os
from unittest.mock import Mock

import pytest

from lightning.fabric.loggers import CSVLogger as FabricCSVLogger
from lightning.fabric.loggers import TensorBoardLogger as FabricTensorBoardLogger
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger


@pytest.fixture
def windows_logger_paths(monkeypatch):
    # Exercise Windows path semantics without changing Python's global os.path.
    windows_os = Mock(wraps=os)
    windows_os.path = ntpath
    for module in (
        "lightning.fabric.utilities.cloud_io",
        "lightning.fabric.loggers.csv_logs",
        "lightning.fabric.loggers.tensorboard",
        "lightning.pytorch.loggers.csv_logs",
        "lightning.pytorch.loggers.tensorboard",
    ):
        monkeypatch.setattr(f"{module}.os", windows_os, raising=False)


@pytest.mark.parametrize("logger_cls", [FabricCSVLogger, FabricTensorBoardLogger, CSVLogger, TensorBoardLogger])
@pytest.mark.parametrize("name", ["experiment", ""])
def test_remote_logger_paths_on_windows(windows_logger_paths, logger_cls, name):
    logger = logger_cls("memory://logs", name=name, version=7)
    expected = f"memory://logs/{name + '/' if name else ''}version_7"
    assert logger.log_dir == expected


@pytest.mark.parametrize("logger_cls", [FabricCSVLogger, FabricTensorBoardLogger, CSVLogger, TensorBoardLogger])
def test_local_logger_paths_on_windows(windows_logger_paths, logger_cls):
    logger = logger_cls(r"C:\logs", name="experiment", version=7)
    assert logger.log_dir == r"C:\logs\experiment\version_7"


@pytest.mark.parametrize("logger_cls", [FabricTensorBoardLogger, TensorBoardLogger])
def test_remote_tensorboard_subdirectory_on_windows(windows_logger_paths, logger_cls):
    logger = logger_cls("memory://logs", name="experiment", version="run", sub_dir="group/nested")
    assert logger.log_dir == "memory://logs/experiment/run/group/nested"


@pytest.mark.parametrize("logger_cls", [FabricCSVLogger, FabricTensorBoardLogger, CSVLogger, TensorBoardLogger])
def test_remote_logger_version_on_windows(windows_logger_paths, logger_cls, tmp_path):
    root = f"memory://{tmp_path.name}/{logger_cls.__module__}"
    logger = logger_cls(root, name="experiment")
    logger._fs.makedirs(f"{root}/experiment/version_3", exist_ok=True)
    assert logger.version == 4


def test_remote_csv_files_on_windows(windows_logger_paths, tmp_path):
    root = f"memory://{tmp_path.name}"
    logger = CSVLogger(root, name="experiment", version=0)
    logger.log_metrics({"loss": 0.5}, step=0)
    logger.log_hyperparams({"batch_size": 2})
    logger.save()
    assert logger._fs.isfile(f"{root}/experiment/version_0/metrics.csv")
    assert logger._fs.isfile(f"{root}/experiment/version_0/hparams.yaml")
