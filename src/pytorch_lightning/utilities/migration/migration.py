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
"""Contains migration functions to upgrade legacy checkpoints to the format of the current Lightning version.

When Lightning loads a checkpoint, these migrations will be applied on the loaded checkpoint dictionary sequentially,
see :func:`~pytorch_lightning.utilities.migration.utils.migrate_checkpoint`.

For the Lightning developer: How to add a new migration?

1. Create a new function with a descriptive name and docstring that explains the details of this migration. Include
   version information as well as the specific commit or PR where the breaking change happened.
2. Add the function to the `_migration_index()` below. The key in the index is the version of Lightning in which the
   change happened. Any checkpoint with a version greater or equal to that version will apply the given function.
   Multiple migrations per version get executed in the provided list order.
3. You can test the migration on a checkpoint (backup your files first) by running:

   cp model.ckpt model.ckpt.backup
   python -m pytorch_lightning.utilities.upgrade_checkpoint --file model.ckpt
"""

from typing import Any, Callable, Dict, List

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

_CHECKPOINT = Dict[str, Any]


def _migration_index() -> Dict[str, List[Callable[[_CHECKPOINT], _CHECKPOINT]]]:
    """Migration functions returned here will get executed in the order they are listed."""
    return {
        "0.10.0": [_migrate_model_checkpoint_early_stopping],
        "1.6.0": [_migrate_loop_global_step_to_progress_tracking, _migrate_loop_current_epoch_to_progress_tracking],
        "1.6.5": [_migrate_loop_batches_that_stepped],
    }


def _migrate_model_checkpoint_early_stopping(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """The checkpoint and early stopping keys were renamed.

    Version: 0.10.0
    Commit: a5d1176
    """
    keys_mapping = {
        "checkpoint_callback_best_model_score": (ModelCheckpoint, "best_model_score"),
        "checkpoint_callback_best_model_path": (ModelCheckpoint, "best_model_path"),
        "checkpoint_callback_best": (ModelCheckpoint, "best_model_score"),
        "early_stop_callback_wait": (EarlyStopping, "wait_count"),
        "early_stop_callback_patience": (EarlyStopping, "patience"),
    }
    checkpoint["callbacks"] = checkpoint.get("callbacks") or {}

    for key, new_path in keys_mapping.items():
        if key in checkpoint:
            value = checkpoint[key]
            callback_type, callback_key = new_path
            checkpoint["callbacks"][callback_type] = checkpoint["callbacks"].get(callback_type) or {}
            checkpoint["callbacks"][callback_type][callback_key] = value
            del checkpoint[key]
    return checkpoint


def _migrate_loop_global_step_to_progress_tracking(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Sets the `global_step` value for checkpoints before v1.6 without the progress tracking state. It will be
    overwritten by the loop's state if it was also saved.

    Version: 1.6.0
    Commit: c67b075
    PR: #13645, #11805
    """
    global_step = checkpoint["global_step"]
    checkpoint.setdefault("loops", {"fit_loop": _FIT_LOOP_INITIAL_STATE_1_6_0})
    checkpoint["loops"].setdefault("fit_loop", _FIT_LOOP_INITIAL_STATE_1_6_0)
    # for automatic optimization
    optim_progress = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_loop.optimizer_loop.optim_progress"]
    optim_progress["optimizer"]["step"]["total"]["completed"] = global_step
    # for manual optimization
    optim_step_progress = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_loop.manual_loop.optim_step_progress"]
    optim_step_progress["total"]["completed"] = global_step
    return checkpoint


def _migrate_loop_current_epoch_to_progress_tracking(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Sets the `current_epoch` value for checkpoints before v1.6 without the progress tracking state. It will be
    overwritten by the loop's state if it was also saved.

    Version: 1.6.0
    Commit: aea96e4
    PR: #11805
    """
    epoch = checkpoint["epoch"]
    checkpoint.setdefault("loops", {"fit_loop": _FIT_LOOP_INITIAL_STATE_1_6_0})
    checkpoint["loops"].setdefault("fit_loop", _FIT_LOOP_INITIAL_STATE_1_6_0)
    checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] = epoch
    return checkpoint


def _migrate_loop_batches_that_stepped(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Sets the `_batches_that_stepped` default value for checkpoints before v1.6.5 which don't have this key.

    Version: 1.6.5
    Commit: c67b075
    PR: #13645
    """
    global_step = checkpoint["global_step"]
    checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"].setdefault("_batches_that_stepped", global_step)
    return checkpoint


_FIT_LOOP_INITIAL_STATE_1_6_0 = {
    "epoch_loop.batch_loop.manual_loop.optim_step_progress": {
        "current": {"completed": 0, "ready": 0},
        "total": {"completed": 0, "ready": 0},
    },
    "epoch_loop.batch_loop.manual_loop.state_dict": {},
    "epoch_loop.batch_loop.optimizer_loop.optim_progress": {
        "optimizer": {
            "step": {"current": {"completed": 0, "ready": 0}, "total": {"completed": 0, "ready": 0}},
            "zero_grad": {
                "current": {"completed": 0, "ready": 0, "started": 0},
                "total": {"completed": 0, "ready": 0, "started": 0},
            },
        },
        "optimizer_position": 0,
    },
    "epoch_loop.batch_loop.optimizer_loop.state_dict": {},
    "epoch_loop.batch_loop.state_dict": {},
    "epoch_loop.batch_progress": {
        "current": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
        "is_last_batch": False,
        "total": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
    },
    "epoch_loop.scheduler_progress": {"current": {"completed": 0, "ready": 0}, "total": {"completed": 0, "ready": 0}},
    "epoch_loop.state_dict": {"_batches_that_stepped": 0},
    "epoch_loop.val_loop.dataloader_progress": {
        "current": {"completed": 0, "ready": 0},
        "total": {"completed": 0, "ready": 0},
    },
    "epoch_loop.val_loop.epoch_loop.batch_progress": {
        "current": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
        "is_last_batch": False,
        "total": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
    },
    "epoch_loop.val_loop.epoch_loop.state_dict": {},
    "epoch_loop.val_loop.state_dict": {},
    "epoch_progress": {
        "current": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
        "total": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
    },
    "state_dict": {},
}
