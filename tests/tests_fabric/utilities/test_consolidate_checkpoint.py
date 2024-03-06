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
from argparse import Namespace
from pathlib import Path
from unittest import mock

import lightning.fabric
import pytest
from lightning.fabric.utilities.consolidate_checkpoint import _parse_cli_args, _process_cli_args
from lightning.fabric.utilities.load import _METADATA_FILENAME


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["path/to/checkpoint"], {"checkpoint_folder": "path/to/checkpoint", "output_file": None}),
        (
            ["path/to/checkpoint", "--output_file", "path/to/output"],
            {"checkpoint_folder": "path/to/checkpoint", "output_file": "path/to/output"},
        ),
    ],
)
def test_parse_cli_args(args, expected):
    with mock.patch("sys.argv", ["any.py", *args]):
        args = _parse_cli_args()
    assert vars(args) == expected


def test_process_cli_args(tmp_path, caplog, monkeypatch):
    # PyTorch version < 2.3
    monkeypatch.setattr(lightning.fabric.utilities.consolidate_checkpoint, "_TORCH_GREATER_EQUAL_2_3", False)
    with caplog.at_level(logging.ERROR, logger="lightning.fabric.utilities.consolidate_checkpoint"), pytest.raises(
        SystemExit
    ):
        _process_cli_args(Namespace())
    assert "requires PyTorch >= 2.3." in caplog.text
    caplog.clear()
    monkeypatch.setattr(lightning.fabric.utilities.consolidate_checkpoint, "_TORCH_GREATER_EQUAL_2_3", True)

    # Checkpoint does not exist
    checkpoint_folder = Path("does/not/exist")
    with caplog.at_level(logging.ERROR, logger="lightning.fabric.utilities.consolidate_checkpoint"), pytest.raises(
        SystemExit
    ):
        _process_cli_args(Namespace(checkpoint_folder=checkpoint_folder))
    assert f"checkpoint folder does not exist: {checkpoint_folder}" in caplog.text
    caplog.clear()

    # Checkpoint exists but is not a folder
    file = tmp_path / "checkpoint_file"
    file.touch()
    with caplog.at_level(logging.ERROR, logger="lightning.fabric.utilities.consolidate_checkpoint"), pytest.raises(
        SystemExit
    ):
        _process_cli_args(Namespace(checkpoint_folder=file))
    assert "checkpoint path must be a folder" in caplog.text
    caplog.clear()

    # Checkpoint exists but is not an FSDP checkpoint
    folder = tmp_path / "checkpoint_folder"
    folder.mkdir()
    with caplog.at_level(logging.ERROR, logger="lightning.fabric.utilities.consolidate_checkpoint"), pytest.raises(
        SystemExit
    ):
        _process_cli_args(Namespace(checkpoint_folder=folder))
    assert "Only FSDP-sharded checkpoints saved with Lightning are supported" in caplog.text
    caplog.clear()

    # Checkpoint is a FSDP folder, output file not specified
    (folder / _METADATA_FILENAME).touch()
    config = _process_cli_args(Namespace(checkpoint_folder=folder, output_file=None))
    assert vars(config) == {
        "checkpoint_folder": folder,
        "output_file": folder.with_suffix(folder.suffix + ".consolidated"),
    }

    # Checkpoint is a FSDP folder, output file already exists
    file = tmp_path / "ouput_file"
    file.touch()
    with caplog.at_level(logging.ERROR, logger="lightning.fabric.utilities.consolidate_checkpoint"), pytest.raises(
        SystemExit
    ):
        _process_cli_args(Namespace(checkpoint_folder=folder, output_file=file))
    assert "path for the converted checkpoint already exists" in caplog.text
    caplog.clear()
