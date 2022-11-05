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
import logging
from pathlib import Path
from unittest import mock
from unittest.mock import ANY

import pytest

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.migration import migrate_checkpoint
from pytorch_lightning.utilities.migration.utils import _get_version, _set_legacy_version, _set_version
from pytorch_lightning.utilities.upgrade_checkpoint import main as upgrade_main


@pytest.mark.parametrize(
    "old_checkpoint, new_checkpoint",
    [
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best": 0.34},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_score": 0.34}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best_model_score": 0.99},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_score": 0.99}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best_model_path": "path"},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_path": "path"}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "early_stop_callback_wait": 2, "early_stop_callback_patience": 4},
            {"epoch": 1, "global_step": 23, "callbacks": {EarlyStopping: {"wait_count": 2, "patience": 4}}},
        ),
    ],
)
def test_upgrade_checkpoint(tmpdir, old_checkpoint, new_checkpoint):
    _set_version(old_checkpoint, "0.9.0")
    _set_legacy_version(new_checkpoint, "0.9.0")
    _set_version(new_checkpoint, pl.__version__)
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint)
    assert updated_checkpoint == old_checkpoint == new_checkpoint
    assert _get_version(updated_checkpoint) == pl.__version__


def test_upgrade_checkpoint_file_missing(tmp_path, caplog):
    # path to single file (missing)
    file = tmp_path / "checkpoint.ckpt"
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(file)]):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                upgrade_main()
            assert f"The path {file} does not exist" in caplog.text

    caplog.clear()

    # path to non-empty directory, but no checkpoints with matching extension
    file.touch()
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(tmp_path), "--extension", ".other"]):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                upgrade_main()
            assert "No checkpoint files with extension .other were found" in caplog.text


@mock.patch("pytorch_lightning.utilities.upgrade_checkpoint.torch.save")
@mock.patch("pytorch_lightning.utilities.upgrade_checkpoint.torch.load")
@mock.patch("pytorch_lightning.utilities.upgrade_checkpoint.migrate_checkpoint")
def test_upgrade_checkpoint_single_file(migrate_mock, load_mock, save_mock, tmp_path):
    file = tmp_path / "checkpoint.ckpt"
    file.touch()
    with mock.patch("sys.argv", ["upgrade_checkpoint.py", str(file)]):
        upgrade_main()

    load_mock.assert_called_once_with(Path(file))
    migrate_mock.assert_called_once()
    save_mock.assert_called_once_with(ANY, Path(file))


@mock.patch("pytorch_lightning.utilities.upgrade_checkpoint.torch.save")
@mock.patch("pytorch_lightning.utilities.upgrade_checkpoint.torch.load")
@mock.patch("pytorch_lightning.utilities.upgrade_checkpoint.migrate_checkpoint")
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
