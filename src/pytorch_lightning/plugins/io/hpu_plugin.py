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

import os
from typing import Any, Dict, Optional

import torch

from lightning_fabric.plugins import TorchCheckpointIO
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.cloud_io import _atomic_save, get_filesystem
from lightning_fabric.utilities.types import _PATH


class HPUCheckpointIO(TorchCheckpointIO):
    """CheckpointIO to save checkpoints for HPU training strategies."""

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``XLACheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = move_data_to_device(checkpoint, torch.device("cpu"))
        # write the checkpoint dictionary to the provided path
        _atomic_save(checkpoint, path)
