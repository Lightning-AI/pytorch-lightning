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
see :func:`migrate_checkpoint`.
"""

import sys
from distutils.version import LooseVersion
from types import ModuleType, TracebackType
from typing import Any, Dict, Optional, Type

import pytorch_lightning as pl
import pytorch_lightning.utilities.argparse

_CHECKPOINT = Dict[str, Any]


class pl_legacy_patch:
    """Registers legacy artifacts (classes, methods, etc.) that were removed but still need to be included for
    unpickling old checkpoints. The following patches apply.

        1. ``pytorch_lightning.utilities.argparse._gpus_arg_default``: Applies to all checkpoints saved prior to
           version 1.2.8. See: https://github.com/PyTorchLightning/pytorch-lightning/pull/6898
        2. ``pytorch_lightning.utilities.argparse_utils``: A module that was deprecated in 1.2 and removed in 1.4,
           but still needs to be available for import for legacy checkpoints.

    Example:

        with pl_legacy_patch():
            torch.load("path/to/legacy/checkpoint.ckpt")
    """

    def __enter__(self) -> "pl_legacy_patch":
        # `pl.utilities.argparse_utils` was renamed to `pl.utilities.argparse`
        legacy_argparse_module = ModuleType("pytorch_lightning.utilities.argparse_utils")
        sys.modules["pytorch_lightning.utilities.argparse_utils"] = legacy_argparse_module

        # `_gpus_arg_default` used to be imported from these locations
        legacy_argparse_module._gpus_arg_default = lambda x: x
        pytorch_lightning.utilities.argparse._gpus_arg_default = lambda x: x
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        if hasattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default"):
            delattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default")
        del sys.modules["pytorch_lightning.utilities.argparse_utils"]


def get_version(checkpoint: _CHECKPOINT) -> str:
    """Get the version of a Lightning checkpoint."""
    return checkpoint["pytorch-lightning_version"]


def set_version(checkpoint: _CHECKPOINT, version: str) -> None:
    """Set the version of a Lightning checkpoint."""
    checkpoint["pytorch-lightning_version"] = version


def should_upgrade(checkpoint: _CHECKPOINT, target: str) -> bool:
    """Returns whether a checkpoint qualifies for an upgrade when the version is lower than the given target."""
    return LooseVersion(get_version(checkpoint)) < LooseVersion(target)


def migrate_checkpoint(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Applies all migrations below in order."""
    if should_upgrade(checkpoint, "0.10.0"):
        _migrate_model_checkpoint_early_stopping(checkpoint)
    if should_upgrade(checkpoint, "1.6.0"):
        _migrate_loop_global_step_to_progress_tracking(checkpoint)
        _migrate_loop_current_epoch_to_progress_tracking(checkpoint)

    set_version(checkpoint, pl.__version__)

    # TODO: If any migrations apply, log a message. Suggest to run upgrade_checkpoint script to convert
    #   checkpoints permanently
    return checkpoint


def _migrate_model_checkpoint_early_stopping(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """The checkpoint and early stopping keys were renamed.

    Version: 0.10.0
    Commit:
    """
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

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
    """Set the `global_step` value for checkpoints before v1.6 without the progress tracking state. It will be
    overwritten by the loop's state if it was also saved.

    Version: 1.6.0
    Commit:
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
    """Set the `current_epoch` value for checkpoints before v1.6 without the progress tracking state. It will be
    overwritten by the loop's state if it was also saved.

    Version: 1.6.0
    Commit:
    """
    epoch = checkpoint["epoch"]
    checkpoint.setdefault("loops", {"fit_loop": _FIT_LOOP_INITIAL_STATE_1_6_0})
    checkpoint["loops"].setdefault("fit_loop", _FIT_LOOP_INITIAL_STATE_1_6_0)
    checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] = epoch


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
