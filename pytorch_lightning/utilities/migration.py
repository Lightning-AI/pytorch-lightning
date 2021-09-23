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
    set_version(checkpoint, pl.__version__)
    return checkpoint


# v0.10.0
def _migrate_model_checkpoint_early_stopping(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
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
