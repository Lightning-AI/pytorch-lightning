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

import sys
import pytorch_lightning as pl
from distutils.version import LooseVersion
from types import ModuleType, TracebackType
from typing import Optional, Type, Dict, Any

from pytorch_lightning.utilities.migration.migrations import migration_index

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


def _get_version(checkpoint: _CHECKPOINT) -> str:
    """Get the version of a Lightning checkpoint."""
    return checkpoint["pytorch-lightning_version"]


def _set_version(checkpoint: _CHECKPOINT, version: str) -> None:
    """Set the version of a Lightning checkpoint."""
    checkpoint["pytorch-lightning_version"] = version


def _should_upgrade(checkpoint: _CHECKPOINT, target: str) -> bool:
    """Returns whether a checkpoint qualifies for an upgrade when the version is lower than the given target."""
    return LooseVersion(_get_version(checkpoint)) < LooseVersion(target)


def migrate_checkpoint(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Applies Lightning version migrations to a checkpoint."""
    index = migration_index()
    for migration_version, migration_functions in index.items():
        if not _should_upgrade(checkpoint, migration_version):
            continue
        for migration_function in migration_functions:
            checkpoint = migration_function(checkpoint)

    _set_version(checkpoint, pl.__version__)

    # TODO: If any migrations apply, log a message. Suggest to run upgrade_checkpoint script to convert
    #   checkpoints permanently
    return checkpoint
