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
