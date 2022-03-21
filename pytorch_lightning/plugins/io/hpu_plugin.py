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

import os
from typing import Any, Dict, Optional

import torch

from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.cloud_io import atomic_save, get_filesystem
from pytorch_lightning.utilities.types import _PATH


class HPUCheckpointIO(TorchCheckpointIO):
    """CheckpointIO to save checkpoints for HPU training strategies."""

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = move_data_to_device(checkpoint, torch.device("cpu"))
        # write the checkpoint dictionary to the provided path
        atomic_save(checkpoint, path)
