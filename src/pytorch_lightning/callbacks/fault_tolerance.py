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
r"""
Fault-Tolerance
^^^^^^^^^^^^^^^

Contains callbacks for fault-tolerance support. These are not meant to be used publicly.
"""
import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.utilities.types import _PATH


class _FaultToleranceCheckpoint(Checkpoint):
    """Used to save a fault-tolerance checkpoint on exception."""

    FILE_EXTENSION = ".ckpt"

    def __init__(self, dirpath: _PATH, filename: str = ".pl_auto_save") -> None:
        super().__init__()
        # not optional because an exception could occur at any moment, so we cannot wait until the `setup` hook
        self.dirpath = dirpath
        if not filename:
            raise ValueError("The filename cannot be empty")
        self.filename = filename

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.dirpath, self.filename + self.FILE_EXTENSION)

    def on_exception(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        # overwrite if necessary
        trainer.save_checkpoint(self.ckpt_path)

    def teardown(self, trainer: "pl.Trainer", *_: Any, **__: Any) -> None:
        trainer.strategy.remove_checkpoint(self.ckpt_path)
