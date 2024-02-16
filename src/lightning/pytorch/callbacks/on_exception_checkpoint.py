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
"""
On exception checkpointing
==========================

Automatically save a checkpoints on exception.
"""

import os
from typing import Any

from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import Checkpoint


class OnExceptionCheckpoint(Checkpoint):
    """Used to save a checkpoint on exception.

    Args:
        dirpath: directory to save the checkpoint file.
        filename: checkpoint filename. This must not include the extension.

    Raises:
        ValueError:
            If ``filename`` is empty.


    Example:
        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import OnExceptionCheckpoint
        >>> trainer = Trainer(callbacks=[OnExceptionCheckpoint(".")])

    """

    FILE_EXTENSION = ".ckpt"

    def __init__(self, dirpath: _PATH, filename: str = "on_exception") -> None:
        super().__init__()
        if not filename:
            raise ValueError("The filename cannot be empty")
        # not optional because an exception could occur at any moment, so we cannot wait until the `setup` hook
        self.dirpath = dirpath
        self.filename = filename

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.dirpath, self.filename + self.FILE_EXTENSION)

    @override
    def on_exception(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        # overwrite if necessary
        trainer.save_checkpoint(self.ckpt_path)

    @override
    def teardown(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        trainer.strategy.remove_checkpoint(self.ckpt_path)
