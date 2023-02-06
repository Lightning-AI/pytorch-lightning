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
import os
import sys
from types import ModuleType, TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type

from packaging.version import Version

import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.types import _PATH
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.utilities.migration.migration import _migration_index
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

_log = logging.getLogger(__name__)
_CHECKPOINT = Dict[str, Any]


def migrate_checkpoint(
    checkpoint: _CHECKPOINT, target_version: Optional[str] = None
) -> Tuple[_CHECKPOINT, Dict[str, List[str]]]:
    """Applies Lightning version migrations to a checkpoint dictionary.

    Args:
        checkpoint: A dictionary with the loaded state from the checkpoint file.
        target_version: Run migrations only up to this version (inclusive), even if migration index contains
            migration functions for newer versions than this target. Mainly useful for testing.

    Note:
        The migration happens in-place. We specifically avoid copying the dict to avoid memory spikes for large
        checkpoints and objects that do not support being deep-copied.
    """
    ckpt_version = _get_version(checkpoint)
    if Version(ckpt_version) > Version(pl.__version__):
        rank_zero_warn(
            f"The loaded checkpoint was produced with Lightning v{ckpt_version}, which is newer than your current"
            f" Lightning version: v{pl.__version__}",
            category=PossibleUserWarning,
        )
        return checkpoint, {}

    index = _migration_index()
    applied_migrations = {}
    for migration_version, migration_functions in index.items():
        if not _should_upgrade(checkpoint, migration_version, target_version):
            continue
        for migration_function in migration_functions:
            checkpoint = migration_function(checkpoint)

        applied_migrations[migration_version] = [fn.__name__ for fn in migration_functions]

    if ckpt_version != pl.__version__:
        _set_legacy_version(checkpoint, ckpt_version)
    _set_version(checkpoint, pl.__version__)
    return checkpoint, applied_migrations


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
        pl.utilities.argparse._gpus_arg_default = lambda x: x
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        if hasattr(pl.utilities.argparse, "_gpus_arg_default"):
            delattr(pl.utilities.argparse, "_gpus_arg_default")
        del sys.modules["pytorch_lightning.utilities.argparse_utils"]


def _pl_migrate_checkpoint(checkpoint: _CHECKPOINT, checkpoint_path: Optional[_PATH] = None) -> _CHECKPOINT:
    """Applies Lightning version migrations to a checkpoint dictionary and prints infos for the user.

    This function is used by the Lightning Trainer when resuming from a checkpoint.
    """
    old_version = _get_version(checkpoint)
    checkpoint, migrations = migrate_checkpoint(checkpoint)
    new_version = _get_version(checkpoint)
    if not migrations or checkpoint_path is None:
        # the checkpoint was already a new one, no migrations were needed
        return checkpoint

    # include the full upgrade command, including the path to the loaded file in the error message,
    # so user can copy-paste and run if they want
    if not _IS_WINDOWS:  # side-step bug: ValueError: path is on mount 'C:', start on mount 'D:'
        path_hint = os.path.relpath(checkpoint_path, os.getcwd())
    else:
        path_hint = os.path.abspath(checkpoint_path)
    _log.info(
        f"Lightning automatically upgraded your loaded checkpoint from v{old_version} to v{new_version}."
        " To apply the upgrade to your files permanently, run"
        f" `python -m pytorch_lightning.utilities.upgrade_checkpoint --file {str(path_hint)}`"
    )
    return checkpoint


def _get_version(checkpoint: _CHECKPOINT) -> str:
    """Get the version of a Lightning checkpoint."""
    return checkpoint["pytorch-lightning_version"]


def _set_version(checkpoint: _CHECKPOINT, version: str) -> None:
    """Set the version of a Lightning checkpoint."""
    checkpoint["pytorch-lightning_version"] = version


def _set_legacy_version(checkpoint: _CHECKPOINT, version: str) -> None:
    """Set the legacy version of a Lightning checkpoint if a legacy version is not already set."""
    checkpoint.setdefault("legacy_pytorch-lightning_version", version)


def _should_upgrade(checkpoint: _CHECKPOINT, target: str, max_version: Optional[str] = None) -> bool:
    """Returns whether a checkpoint qualifies for an upgrade when the version is lower than the given target."""
    is_lte_max_version = max_version is None or Version(target) <= Version(max_version)
    return Version(_get_version(checkpoint)) < Version(target) and is_lte_max_version
