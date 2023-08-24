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

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger


class TestModel(BoringModel):
    def __init__(self, expected_log_dir):
        super().__init__()
        self.expected_log_dir = expected_log_dir

    def training_step(self, *args, **kwargs):
        assert self.trainer.log_dir == self.expected_log_dir
        return super().training_step(*args, **kwargs)


def test_logdir(tmp_path):
    """Tests that the path is correct when checkpoint and loggers are used."""
    expected = os.path.join(tmp_path, "lightning_logs", "version_0")

    model = TestModel(expected)

    trainer = Trainer(default_root_dir=tmp_path, max_steps=2, callbacks=[ModelCheckpoint(dirpath=tmp_path)])

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_no_checkpoint_cb(tmp_path):
    """Tests that the path is correct with no checkpoint."""
    expected = os.path.join(tmp_path, "lightning_logs", "version_0")
    model = TestModel(expected)

    trainer = Trainer(default_root_dir=tmp_path, max_steps=2, enable_checkpointing=False)

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_no_logger(tmp_path):
    """Tests that the path is correct even when there is no logger."""
    expected = os.path.join(tmp_path)
    model = TestModel(expected)

    trainer = Trainer(
        default_root_dir=tmp_path, max_steps=2, logger=False, callbacks=[ModelCheckpoint(dirpath=tmp_path)]
    )

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_no_logger_no_checkpoint(tmp_path):
    """Tests that the path is correct even when there is no logger."""
    expected = os.path.join(tmp_path)
    model = TestModel(expected)

    trainer = Trainer(default_root_dir=tmp_path, max_steps=2, logger=False, enable_checkpointing=False)

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_custom_callback(tmp_path):
    """Tests that the path is correct even when there is a custom callback."""
    expected = os.path.join(tmp_path, "lightning_logs", "version_0")
    model = TestModel(expected)

    trainer = Trainer(
        default_root_dir=tmp_path, max_steps=2, callbacks=[ModelCheckpoint(dirpath=os.path.join(tmp_path, "ckpts"))]
    )

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_custom_logger(tmp_path):
    """Tests that the path is correct even when there is a custom logger."""
    expected = os.path.join(tmp_path, "custom_logs", "version_0")
    model = TestModel(expected)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=2,
        callbacks=[ModelCheckpoint(dirpath=tmp_path)],
        logger=TensorBoardLogger(save_dir=tmp_path, name="custom_logs"),
    )

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected


def test_logdir_multiple_loggers(tmp_path):
    """Tests that the logdir equals the default_root_dir when trainer has multiple loggers."""
    default_root_dir = tmp_path / "default_root_dir"
    save_dir = tmp_path / "save_dir"
    expected = os.path.join(tmp_path, "save_dir", "custom_logs", "version_0")
    model = TestModel(expected)
    trainer = Trainer(
        default_root_dir=default_root_dir,
        max_steps=2,
        logger=[TensorBoardLogger(save_dir=save_dir, name="custom_logs"), CSVLogger(tmp_path)],
    )

    assert trainer.log_dir == expected
    trainer.fit(model)
    assert trainer.log_dir == expected
