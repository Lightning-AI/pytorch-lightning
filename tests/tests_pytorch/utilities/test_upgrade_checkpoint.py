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
from pathlib import Path
from unittest import mock
from unittest.mock import ANY

import pytest
import torch
from lightning.pytorch.utilities.upgrade_checkpoint import main as upgrade_main


def test_upgrade_checkpoint_file_missing(tmp_path, caplog):
    # path to single file (missing)
    file = tmp_path / "checkpoint.ckpt"
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(file)]), caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            upgrade_main()
        assert f"The path {file} does not exist" in caplog.text

    caplog.clear()

    # path to non-empty directory, but no checkpoints with matching extension
    file.touch()
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(tmp_path), "--extension", ".other"]), caplog.at_level(
        logging.ERROR
    ):
        with pytest.raises(SystemExit):
            upgrade_main()
        assert "No checkpoint files with extension .other were found" in caplog.text


@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.torch.save")
@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.torch.load")
@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.migrate_checkpoint")
def test_upgrade_checkpoint_single_file(migrate_mock, load_mock, save_mock, tmp_path):
    file = tmp_path / "checkpoint.ckpt"
    file.touch()
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(file)]):
        upgrade_main()

    load_mock.assert_called_once_with(Path(file), map_location=None)
    migrate_mock.assert_called_once()
    save_mock.assert_called_once_with(ANY, Path(file))


@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.torch.save")
@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.torch.load")
@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.migrate_checkpoint")
def test_upgrade_checkpoint_directory(migrate_mock, load_mock, save_mock, tmp_path):
    top_files = [tmp_path / "top0.ckpt", tmp_path / "top1.ckpt"]
    nested_files = [
        tmp_path / "subdir0" / "nested0.ckpt",
        tmp_path / "subdir0" / "nested1.other",
        tmp_path / "subdir1" / "nested2.ckpt",
    ]

    for file in top_files + nested_files:
        file.parent.mkdir(exist_ok=True, parents=True)
        file.touch()

    # directory with recursion
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(tmp_path)]):
        upgrade_main()

    assert {c[0][0] for c in load_mock.call_args_list} == {
        tmp_path / "top0.ckpt",
        tmp_path / "top1.ckpt",
        tmp_path / "subdir0" / "nested0.ckpt",
        tmp_path / "subdir1" / "nested2.ckpt",
    }
    assert migrate_mock.call_count == 4
    assert {c[0][1] for c in save_mock.call_args_list} == {
        tmp_path / "top0.ckpt",
        tmp_path / "top1.ckpt",
        tmp_path / "subdir0" / "nested0.ckpt",
        tmp_path / "subdir1" / "nested2.ckpt",
    }


@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.torch.load")
@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.torch.save")
@mock.patch("lightning.pytorch.utilities.upgrade_checkpoint.migrate_checkpoint")
def test_upgrade_checkpoint_map_location(_, __, load_mock, tmp_path):
    file = tmp_path / "checkpoint.ckpt"
    file.touch()
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(file)]):
        upgrade_main()
    assert load_mock.call_args[1]["map_location"] is None
    load_mock.reset_mock()
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(file), "--map-to-cpu"]):
        upgrade_main()
    assert load_mock.call_args[1]["map_location"] == torch.device("cpu")
