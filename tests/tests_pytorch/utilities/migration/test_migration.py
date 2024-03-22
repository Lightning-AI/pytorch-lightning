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
from unittest.mock import ANY, MagicMock

import lightning.pytorch as pl
import pytest
import torch
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, ManualOptimBoringModel
from lightning.pytorch.utilities.migration import migrate_checkpoint
from lightning.pytorch.utilities.migration.utils import _get_version, _set_legacy_version, _set_version


@pytest.mark.parametrize(
    ("old_checkpoint", "new_checkpoint"),
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
            {
                "epoch": 1,
                "global_step": 23,
                "callbacks": {EarlyStopping: {"wait_count": 2, "patience": 4}},
            },
        ),
    ],
)
def test_migrate_model_checkpoint_early_stopping(old_checkpoint, new_checkpoint):
    _set_version(old_checkpoint, "0.9.0")
    _set_legacy_version(new_checkpoint, "0.9.0")
    _set_version(new_checkpoint, pl.__version__)
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint, target_version="1.0.0")
    assert updated_checkpoint == old_checkpoint == new_checkpoint
    assert _get_version(updated_checkpoint) == pl.__version__


def test_migrate_loop_global_step_to_progress_tracking():
    old_checkpoint = {"global_step": 15, "epoch": 2}
    _set_version(old_checkpoint, "1.5.9")  # pretend a checkpoint prior to 1.6.0
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint, target_version="1.6.0")
    # automatic optimization
    assert (
        updated_checkpoint["loops"]["fit_loop"]["epoch_loop.batch_loop.optimizer_loop.optim_progress"]["optimizer"][
            "step"
        ]["total"]["completed"]
        == 15
    )
    # for manual optimization
    assert (
        updated_checkpoint["loops"]["fit_loop"]["epoch_loop.batch_loop.manual_loop.optim_step_progress"]["total"][
            "completed"
        ]
        == 15
    )


def test_migrate_loop_current_epoch_to_progress_tracking():
    old_checkpoint = {"global_step": 15, "epoch": 2}
    _set_version(old_checkpoint, "1.5.9")  # pretend a checkpoint prior to 1.6.0
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint)
    assert updated_checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == 2


@pytest.mark.parametrize("model_class", [BoringModel, ManualOptimBoringModel])
def test_migrate_loop_batches_that_stepped(tmp_path, model_class):
    trainer = Trainer(max_steps=1, limit_val_batches=0, default_root_dir=tmp_path)
    model = model_class()
    trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path

    # pretend we have a checkpoint produced in < v1.6.5; the key "_batches_that_stepped" didn't exist back then
    ckpt = torch.load(ckpt_path)
    del ckpt["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"]
    _set_version(ckpt, "1.6.4")
    torch.save(ckpt, ckpt_path)

    class TestModel(model_class):
        def on_train_start(self) -> None:
            assert self.trainer.global_step == 1
            assert self.trainer.fit_loop.epoch_loop._batches_that_stepped == 1

    trainer = Trainer(max_steps=2, limit_val_batches=0, default_root_dir=tmp_path)
    model = TestModel()
    trainer.fit(model, ckpt_path=ckpt_path)
    new_loop = trainer.fit_loop.epoch_loop
    assert new_loop.global_step == new_loop._batches_that_stepped == 2


@pytest.mark.parametrize("save_on_train_epoch_end", [None, True, False])
def test_migrate_model_checkpoint_save_on_train_epoch_end_default(save_on_train_epoch_end):
    """Test that the 'save_on_train_epoch_end' part of the ModelCheckpoint state key gets removed."""
    legacy_state_key = (
        f"ModelCheckpoint{{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        f" 'train_time_interval': None, 'save_on_train_epoch_end': {save_on_train_epoch_end}}}"
    )
    new_state_key = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        " 'train_time_interval': None}"
    )
    old_checkpoint = {"callbacks": {legacy_state_key: {"dummy": 0}}, "global_step": 0, "epoch": 1}
    _set_version(old_checkpoint, "1.8.9")  # pretend a checkpoint prior to 1.9.0
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint, target_version="1.9.0")
    assert updated_checkpoint["callbacks"] == {new_state_key: {"dummy": 0}}  # None -> None


def test_migrate_model_checkpoint_save_on_train_epoch_end_default_collision():
    """Test that the migration warns about collisions that would occur if the keys were modified."""
    # The two keys only differ in the `save_on_train_epoch_end` value
    state_key1 = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        " 'train_time_interval': None, 'save_on_train_epoch_end': True}"
    )
    state_key2 = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        " 'train_time_interval': None, 'save_on_train_epoch_end': False}"
    )
    old_checkpoint = {
        "callbacks": {state_key1: {"dummy": 0}, state_key2: {"dummy": 0}},
        "global_step": 0,
        "epoch": 1,
    }
    _set_version(old_checkpoint, "1.8.9")  # pretend a checkpoint prior to 1.9.0
    with pytest.warns(PossibleUserWarning, match="callback states in this checkpoint.* colliding with each other"):
        updated_checkpoint, _ = migrate_checkpoint(old_checkpoint.copy(), target_version="1.9.0")
    assert updated_checkpoint["callbacks"] == old_checkpoint["callbacks"]  # no migration was performed


def test_migrate_dropped_apex_amp_state(monkeypatch):
    """Test that the migration warns about collisions that would occur if the keys were modified."""
    monkeypatch.setattr(pl, "__version__", "2.0.0")  # pretend this version of Lightning is >= 2.0.0
    old_checkpoint = {"amp_scaling_state": {"scale": 1.23}}
    _set_version(old_checkpoint, "1.9.0")  # pretend a checkpoint prior to 2.0.0
    with pytest.warns(UserWarning, match="checkpoint contains apex AMP data"):
        updated_checkpoint, _ = migrate_checkpoint(old_checkpoint.copy())
    assert "amp_scaling_state" not in updated_checkpoint


def test_migrate_loop_structure_after_tbptt_removal():
    """Test the loop state migration after truncated backpropagation support was removed in 2.0.0, and with it the
    training batch loop."""
    # automatic- and manual optimization state are combined into a single checkpoint to simplify testing
    state_automatic = MagicMock()
    state_manual = MagicMock()
    optim_progress_automatic = MagicMock()
    optim_progress_manual = MagicMock()
    old_batch_loop_state = MagicMock()
    old_checkpoint = {
        "loops": {
            "fit_loop": {
                "epoch_loop.state_dict": {"any": "state"},
                "epoch_loop.batch_loop.state_dict": old_batch_loop_state,
                "epoch_loop.batch_loop.optimizer_loop.state_dict": state_automatic,
                "epoch_loop.batch_loop.optimizer_loop.optim_progress": optim_progress_automatic,
                "epoch_loop.batch_loop.manual_loop.state_dict": state_manual,
                "epoch_loop.batch_loop.manual_loop.optim_step_progress": optim_progress_manual,
            }
        }
    }
    _set_version(old_checkpoint, "1.8.0")  # pretend a checkpoint prior to 2.0.0
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint.copy(), target_version="2.0.0")
    assert updated_checkpoint["loops"] == {
        "fit_loop": {
            "epoch_loop.state_dict": {"any": "state", "old_batch_loop_state_dict": old_batch_loop_state},
            "epoch_loop.automatic_optimization.state_dict": state_automatic,
            "epoch_loop.automatic_optimization.optim_progress": optim_progress_automatic,
            "epoch_loop.manual_optimization.state_dict": state_manual,
            "epoch_loop.manual_optimization.optim_step_progress": optim_progress_manual,
        }
    }


def test_migrate_loop_structure_after_optimizer_loop_removal():
    """Test the loop state migration after multiple optimizer support in automatic optimization was removed in
    2.0.0."""
    state_automatic = MagicMock()
    state_manual = MagicMock()
    optim_progress_automatic = {
        "optimizer": MagicMock(),
        "optimizer_position": 33,
    }
    optim_progress_manual = MagicMock()
    old_checkpoint = {
        "loops": {
            "fit_loop": {
                "epoch_loop.state_dict": {"any": "state"},
                "epoch_loop.batch_loop.state_dict": MagicMock(),
                "epoch_loop.batch_loop.optimizer_loop.state_dict": state_automatic,
                "epoch_loop.batch_loop.optimizer_loop.optim_progress": optim_progress_automatic,
                "epoch_loop.batch_loop.manual_loop.state_dict": state_manual,
                "epoch_loop.batch_loop.manual_loop.optim_step_progress": optim_progress_manual,
            }
        }
    }
    _set_version(old_checkpoint, "1.9.0")  # pretend a checkpoint prior to 2.0.0
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint.copy(), target_version="2.0.0")
    assert updated_checkpoint["loops"] == {
        "fit_loop": {
            "epoch_loop.state_dict": ANY,
            "epoch_loop.automatic_optimization.state_dict": state_automatic,
            "epoch_loop.automatic_optimization.optim_progress": {"optimizer": ANY},  # optimizer_position gets dropped
            "epoch_loop.manual_optimization.state_dict": state_manual,
            "epoch_loop.manual_optimization.optim_step_progress": optim_progress_manual,
        }
    }


def test_migrate_loop_structure_after_dataloader_loop_removal():
    """Test the loop state migration after the dataloader loops were removed in 2.0.0."""
    old_dataloader_loop_state_dict = {
        "state_dict": {},
        "dataloader_progress": {"total": {"ready": 0, "completed": 0}, "current": {"ready": 0, "completed": 0}},
        "epoch_loop.state_dict": {},
        "epoch_loop.batch_progress": {
            "total": {"ready": 123, "started": 0, "processed": 0, "completed": 0},
            "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            "is_last_batch": False,
        },
    }
    old_checkpoint = {
        "loops": {
            "predict_loop": old_dataloader_loop_state_dict,
            "validate_loop": dict(old_dataloader_loop_state_dict),  # copy
            "test_loop": dict(old_dataloader_loop_state_dict),  # copy
        }
    }
    _set_version(old_checkpoint, "1.9.0")  # pretend a checkpoint prior to 2.0.0
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint.copy(), target_version="2.0.0")
    assert updated_checkpoint["loops"] == {
        "predict_loop": {
            "batch_progress": {
                "current": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
                "is_last_batch": False,
                "total": {"completed": 0, "processed": 0, "ready": 123, "started": 0},
            },
            "state_dict": {},
        },
        "test_loop": {
            "batch_progress": {
                "current": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
                "is_last_batch": False,
                "total": {"completed": 0, "processed": 0, "ready": 123, "started": 0},
            },
            "state_dict": {},
        },
        "validate_loop": {
            "batch_progress": {
                "current": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
                "is_last_batch": False,
                "total": {"completed": 0, "processed": 0, "ready": 123, "started": 0},
            },
            "state_dict": {},
        },
    }
