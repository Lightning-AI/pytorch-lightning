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
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.migration.utils import _set_version


def test_preloaded_checkpoint_lifecycle(tmp_path):
    """Tests that the preloaded checkpoint contents gets cleared from memory when it is not required anymore."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1)
    trainer.fit(model)

    connector = trainer._checkpoint_connector

    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint

    connector.resume_start()
    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint
    connector.resume_end()
    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint

    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer = Trainer(default_root_dir=tmp_path, max_steps=2)
    connector = trainer._checkpoint_connector
    connector.resume_start(ckpt_path)
    assert connector._ckpt_path == ckpt_path
    assert connector._loaded_checkpoint
    assert isinstance(connector._loaded_checkpoint, dict)
    trainer.state.fn = TrainerFn.FITTING
    connector.resume_end()
    # not cleared until next restoration, as the user might access it through `trainer.ckpt_path`
    assert connector._ckpt_path == ckpt_path
    assert not connector._loaded_checkpoint


@mock.patch("lightning.fabric.plugins.environments.slurm.SLURMEnvironment.detect", return_value=True)
def test_hpc_restore_attempt(_, tmp_path):
    """Test that restore() attempts to restore the hpc_ckpt with highest priority."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1, enable_checkpointing=False, logger=False)
    trainer.fit(model)

    hpc_ckpt_path = tmp_path / "hpc_ckpt_3.ckpt"
    trainer.save_checkpoint(hpc_ckpt_path)
    assert os.listdir(tmp_path) == ["hpc_ckpt_3.ckpt"]

    # set weights to zero
    for param in model.parameters():
        torch.nn.init.constant_(param, 0)

    # case 1: restore hpc first, no explicit resume path provided
    trainer = Trainer(default_root_dir=tmp_path, max_steps=2, enable_checkpointing=False, logger=False)
    trainer.fit(model)

    for param in model.parameters():
        assert param.abs().sum() > 0
        torch.nn.init.constant_(param, 0)

    # case 2: explicit resume path provided, file not found
    trainer = Trainer(default_root_dir=tmp_path, max_steps=3)

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found: not existing"):
        trainer.fit(model, ckpt_path="not existing")


def test_hpc_max_ckpt_version(tmp_path):
    """Test that the _CheckpointConnector is able to find the hpc checkpoint file with the highest version."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1)
    trainer.fit(model)
    trainer.save_checkpoint(tmp_path / "hpc_ckpt.ckpt")
    trainer.save_checkpoint(tmp_path / "hpc_ckpt_0.ckpt")
    trainer.save_checkpoint(tmp_path / "hpc_ckpt_3.ckpt")
    trainer.save_checkpoint(tmp_path / "hpc_ckpt_33.ckpt")

    assert trainer._checkpoint_connector._hpc_resume_path == str(tmp_path / "hpc_ckpt_33.ckpt")
    assert trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmp_path) == 33
    assert (
        trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmp_path / "not" / "existing")
        is None
    )


def test_ckpt_for_fsspec():
    """Test that the _CheckpointConnector is able to write to fsspec file systems."""
    model = BoringModel()
    # hardcoding dir since `tmp_path` can be windows path
    trainer = Trainer(
        default_root_dir="memory://test_ckpt_for_fsspec", limit_train_batches=1, limit_val_batches=1, max_epochs=1
    )
    trainer.fit(model)
    trainer.save_checkpoint("memory://test_ckpt_for_fsspec/hpc_ckpt.ckpt")
    trainer.save_checkpoint("memory://test_ckpt_for_fsspec/hpc_ckpt_0.ckpt")
    trainer.save_checkpoint("memory://test_ckpt_for_fsspec/hpc_ckpt_3.ckpt")
    trainer.save_checkpoint("memory://test_ckpt_for_fsspec/hpc_ckpt_33.ckpt")

    assert trainer._checkpoint_connector._hpc_resume_path == "memory://test_ckpt_for_fsspec/hpc_ckpt_33.ckpt"
    assert (
        trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder("memory://test_ckpt_for_fsspec")
        == 33
    )
    assert (
        trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder("memory://not_existing") is None
    )


def test_loops_restore(tmp_path):
    """Test that required loop state_dict is loaded correctly by checkpoint connector."""
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    trainer_args = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "logger": False,
        "callbacks": [checkpoint_callback],
        "num_sanity_val_steps": 0,
    }
    trainer = Trainer(**trainer_args)
    trainer.fit(model)

    ckpt_path = str(tmp_path / "last.ckpt")

    trainer = Trainer(**trainer_args)
    trainer.strategy.connect(model)

    trainer_fns = list(TrainerFn)
    for fn in trainer_fns:
        trainer_fn = getattr(trainer, f"{fn.value}_loop")
        trainer_fn.load_state_dict = mock.Mock()

    for fn in trainer_fns:
        trainer.state.fn = fn
        trainer._checkpoint_connector.resume_start(ckpt_path)
        trainer._checkpoint_connector.restore_loops()

        trainer_loop = getattr(trainer, f"{fn.value}_loop")
        trainer_loop.load_state_dict.assert_called()
        trainer_loop.load_state_dict.reset_mock()

        for fn2 in trainer_fns:
            if fn2 != fn:
                trainer_loop2 = getattr(trainer, f"{fn2.value}_loop")
                trainer_loop2.load_state_dict.assert_not_called()


def test_stateful_trainer_ckpt_path_support(tmp_path):
    """Tests support for the pattern used by NeMo's experiment manager."""
    model = BoringModel()

    # dummy ckpt data
    ckpt_data = {"state_dict": model.state_dict(), "optimizer_states": {}, "lr_schedulers": {}}
    _set_version(ckpt_data, "2.0.0")

    # save a "checkpoint"
    ckpt_path = tmp_path / "foo.ckpt"
    torch.save(ckpt_data, ckpt_path)

    # mock model checkpoint instance that has saved a last checkpoint
    model_checkpoint = Mock(spec=ModelCheckpoint)
    last_path = tmp_path / "last.ckpt"
    torch.save(ckpt_data, last_path)
    model_checkpoint._find_last_checkpoints.return_value = {last_path}

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, callbacks=model_checkpoint)

    # set the ckpt path statefully
    trainer.ckpt_path = ckpt_path
    trainer.fit(model)
    assert trainer.ckpt_path == ckpt_path  # not automatically cleaned
    assert trainer._checkpoint_connector._user_managed

    # now conflict with ckpt_path functionally
    with pytest.warns(UserWarning, match="trainer.ckpt_path =.*but then you passed"):
        trainer.fit(model, ckpt_path="last")
    assert trainer.ckpt_path == last_path
    assert not trainer._checkpoint_connector._user_managed

    # mock model checkpoint instance that has saved a last checkpoint
    best_path = tmp_path / "best.ckpt"
    torch.save(ckpt_data, best_path)
    model_checkpoint.best_model_path = best_path

    # `trainer.test` will use this over "best" if statefully set
    trainer.ckpt_path = ckpt_path
    trainer.test()
    assert trainer.ckpt_path == ckpt_path

    # ckpt_path = "best" still works if it's reset
    trainer.ckpt_path = None
    # the state is cleared
    assert trainer._checkpoint_connector._ckpt_path is None
    assert not trainer._checkpoint_connector._user_managed
    trainer.test()
    assert trainer.ckpt_path == best_path


@pytest.mark.parametrize(("strict_loading", "expected"), [(None, True), (True, True), (False, False)])
def test_strict_loading(strict_loading, expected, tmp_path):
    """Test that the connector respects the `LightningModule.strict_loading` setting."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, barebones=True, max_steps=1)
    trainer.fit(model)
    trainer.save_checkpoint(tmp_path / "checkpoint.ckpt")

    model = BoringModel()
    model.strict_loading = strict_loading
    model.load_state_dict = Mock()

    trainer = Trainer(default_root_dir=tmp_path, barebones=True, max_steps=2)
    trainer.fit(model, ckpt_path=(tmp_path / "checkpoint.ckpt"))
    model.load_state_dict.assert_called_once_with(ANY, strict=expected)
