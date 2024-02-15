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
"""Contains migration functions to upgrade legacy checkpoints to the format of the current Lightning version.

When Lightning loads a checkpoint, these migrations will be applied on the loaded checkpoint dictionary sequentially,
see :func:`~lightning.pytorch.utilities.migration.utils.migrate_checkpoint`.

For the Lightning developer: How to add a new migration?

1. Create a new function with a descriptive name and docstring that explains the details of this migration. Include
   version information as well as the specific commit or PR where the breaking change happened.
2. Add the function to the `_migration_index()` below. The key in the index is the version of Lightning in which the
   change happened. Any checkpoint with a version greater or equal to that version will apply the given function.
   Multiple migrations per version get executed in the provided list order.
3. You can test the migration on a checkpoint (backup your files first) by running:

   cp model.ckpt model.ckpt.backup
   python -m lightning.pytorch.utilities.upgrade_checkpoint model.ckpt

"""

import re
from typing import Any, Callable, Dict, List

from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

_CHECKPOINT = Dict[str, Any]


def _migration_index() -> Dict[str, List[Callable[[_CHECKPOINT], _CHECKPOINT]]]:
    """Migration functions returned here will get executed in the order they are listed."""
    return {
        "0.10.0": [_migrate_model_checkpoint_early_stopping],
        "1.6.0": [_migrate_loop_global_step_to_progress_tracking, _migrate_loop_current_epoch_to_progress_tracking],
        "1.6.5": [_migrate_loop_batches_that_stepped],
        "1.9.0": [_migrate_model_checkpoint_save_on_train_epoch_end_default],
        "2.0.0": [
            _drop_apex_amp_state,
            _migrate_loop_structure_after_tbptt_removal,
            _migrate_loop_structure_after_optimizer_loop_removal,
            _migrate_loop_structure_after_dataloader_loop_removal,
        ],
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
    checkpoint.setdefault("loops", {"fit_loop": _get_fit_loop_initial_state_1_6_0()})
    checkpoint["loops"].setdefault("fit_loop", _get_fit_loop_initial_state_1_6_0())
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
    checkpoint.setdefault("loops", {"fit_loop": _get_fit_loop_initial_state_1_6_0()})
    checkpoint["loops"].setdefault("fit_loop", _get_fit_loop_initial_state_1_6_0())
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


def _get_fit_loop_initial_state_1_6_0() -> Dict:
    return {
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
        "epoch_loop.scheduler_progress": {
            "current": {"completed": 0, "ready": 0},
            "total": {"completed": 0, "ready": 0},
        },
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


def _migrate_model_checkpoint_save_on_train_epoch_end_default(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """The ``save_on_train_epoch_end`` was removed from the state-key of ``ModelCheckpoint`` in 1.9.0, and this
    migration drops it from the state-keys saved in the checkpoint dict so that the keys match when the Trainer loads
    the callback state.

    Version: 1.9.0
    Commit: f4ca56
    PR: #15300, #15606

    """
    if "callbacks" not in checkpoint:
        return checkpoint

    def new_key(old_key: str) -> str:
        if not old_key.startswith("ModelCheckpoint"):
            return old_key
        return re.sub(", 'save_on_train_epoch_end': (None|True|False)", "", old_key)

    num_keys = len(checkpoint["callbacks"])
    # Note: only iterate over keys that are strings. The legacy state key was the type of the callback.
    new_callback_states = {
        new_key(old_key): state for old_key, state in checkpoint["callbacks"].items() if isinstance(old_key, str)
    }
    if len(new_callback_states) < num_keys:
        rank_zero_warn(
            "You have multiple `ModelCheckpoint` callback states in this checkpoint, but we found state keys"
            " that would end up colliding with each other after an upgrade, which means we can't differentiate"
            " which of your checkpoint callbacks needs which states. At least one of your `ModelCheckpoint`"
            " callbacks will not be able to reload the state.",
            category=PossibleUserWarning,
        )
        return checkpoint

    checkpoint["callbacks"] = new_callback_states
    return checkpoint


def _drop_apex_amp_state(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Apex support was removed in v2.0.0, and this migration drops it from the state-keys saved in the checkpoint
    dict.

    Version: 2.0.0
    Commit: e544676ff434ed96c6dd3b4e73a708bcb27ebcf1
    PR: #16149

    """
    key = "amp_scaling_state"
    if key in checkpoint:
        rank_zero_warn("This checkpoint contains apex AMP data, but apex support has been removed in v2.0.0.")
        del checkpoint[key]
    return checkpoint


def _migrate_loop_structure_after_tbptt_removal(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Adjusts the loop structure since it changed when the support for truncated backpropagation was removed. The
    optimizer loop and the manual loop were previously children of the training batch loop. After its removal, they
    became the children of the training epoch loop.

    Version: 2.0.0
    Commit: 7807454
    PR: #16337, #16172

    """
    if "loops" not in checkpoint:
        return checkpoint
    if "fit_loop" not in checkpoint["loops"]:
        return checkpoint
    fit_loop = checkpoint["loops"]["fit_loop"]

    # remap `x.batch_loop.y` to `x.y`
    old_key_new_key_mapping = {
        "epoch_loop.batch_loop.manual_loop.optim_step_progress": "epoch_loop.manual_loop.optim_step_progress",
        "epoch_loop.batch_loop.manual_loop.state_dict": "epoch_loop.manual_loop.state_dict",
        "epoch_loop.batch_loop.optimizer_loop.optim_progress": "epoch_loop.optimizer_loop.optim_progress",
        "epoch_loop.batch_loop.optimizer_loop.state_dict": "epoch_loop.optimizer_loop.state_dict",
    }
    for old, new in list(old_key_new_key_mapping.items()):
        if old in fit_loop:
            fit_loop[new] = fit_loop[old]
            del fit_loop[old]

    # We can safely drop this key: our default implementation of `batch_loop` did not have state.
    # If there was state from a custom batch loop, we wouldn't be able to load it meaningfully.
    # But just in case, we save a copy of it in `epoch_loop.state_dict` in case the user wants to process it after
    # loading the checkpoint.
    if "epoch_loop.batch_loop.state_dict" in fit_loop and fit_loop["epoch_loop.batch_loop.state_dict"]:
        fit_loop["epoch_loop.state_dict"]["old_batch_loop_state_dict"] = fit_loop["epoch_loop.batch_loop.state_dict"]
    fit_loop.pop("epoch_loop.batch_loop.state_dict", None)

    return checkpoint


def _migrate_loop_structure_after_optimizer_loop_removal(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Adjusts the loop structure since it changed when the support for multiple optimizers in automatic optimization
    mode was removed. There is no longer a loop over optimizer, and hence no position to store for resuming the loop.

    Version: 2.0.0
    Commit: 6a56586
    PR: #16539, #16598

    """
    if "loops" not in checkpoint:
        return checkpoint
    if "fit_loop" not in checkpoint["loops"]:
        return checkpoint
    fit_loop = checkpoint["loops"]["fit_loop"]

    # optimizer_position is no longer used
    if "epoch_loop.optimizer_loop.optim_progress" in fit_loop:
        fit_loop["epoch_loop.optimizer_loop.optim_progress"].pop("optimizer_position", None)

    # the subloop attribute names have changed
    if "epoch_loop.optimizer_loop.state_dict" in fit_loop:
        fit_loop["epoch_loop.automatic_optimization.state_dict"] = fit_loop.pop("epoch_loop.optimizer_loop.state_dict")
        fit_loop["epoch_loop.automatic_optimization.optim_progress"] = fit_loop.pop(
            "epoch_loop.optimizer_loop.optim_progress"
        )
    if "epoch_loop.manual_loop.state_dict" in fit_loop:
        fit_loop["epoch_loop.manual_optimization.state_dict"] = fit_loop.pop("epoch_loop.manual_loop.state_dict")
        fit_loop["epoch_loop.manual_optimization.optim_step_progress"] = fit_loop.pop(
            "epoch_loop.manual_loop.optim_step_progress"
        )
    return checkpoint


def _migrate_loop_structure_after_dataloader_loop_removal(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """The dataloader loops (``_DataLoaderLoop``, ``_PredictionLoop`, and ``_EvaluationLoop``) were flattened into the
    ``_EvaluationEpochLoop`` (now ``_EvaluationLoop``) and ``_PredictionEpochLoop`` (now ``_PredictionLoop``).

    Version: 2.0.0
    Commit: ec4f592ecfe238edd83185f6c6905fb1e2406d61
    PR: #16726

    """
    if "loops" not in checkpoint:
        return checkpoint
    loops = checkpoint["loops"]
    for loop_key in ("predict_loop", "validate_loop", "test_loop"):
        if loop_key not in loops:
            continue
        loop = loops[loop_key]
        loop.pop("dataloader_progress", None)  # no longer used
        epoch_loop_key = "epoch_loop."
        epoch_loop_dict = {k[len(epoch_loop_key) :]: loop.pop(k) for k in list(loop) if k.startswith(epoch_loop_key)}
        loop.update(epoch_loop_dict)
    return checkpoint
