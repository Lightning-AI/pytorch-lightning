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
from typing import Any, Dict, Optional

from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO


class _WrappingCheckpointIO(CheckpointIO):
    """``_WrappingCheckpointIO`` is a wrapper checkpoint_io that uses a base checkpoint_io to handle checkpointing.

    Args:
        checkpoint_io: A checkpoint IO plugin that is used as the basis.
    """

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None) -> None:
        super().__init__()

        self._checkpoint_io = checkpoint_io
        self._base_checkpoint_io_configured: bool = False

        if checkpoint_io is not None:
            if isinstance(checkpoint_io, _WrappingCheckpointIO):
                self._base_checkpoint_io_configured = checkpoint_io._base_checkpoint_io_configured
            else:
                self._base_checkpoint_io_configured = True

    @property
    def checkpoint_io(self) -> Optional["CheckpointIO"]:
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, checkpoint_io: "CheckpointIO") -> None:
        assert not isinstance(checkpoint_io, _WrappingCheckpointIO)

        if self._checkpoint_io is None:
            self._base_checkpoint_io_configured = True
            self._checkpoint_io = checkpoint_io
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO) and not self._base_checkpoint_io_configured:
            self._base_checkpoint_io_configured = True
            self._checkpoint_io.checkpoint_io = checkpoint_io

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Uses the base ``checkpoint_io`` to save the checkpoint."""
        assert self.checkpoint_io is not None
        self.checkpoint_io.save_checkpoint(*args, **kwargs)

    def remove_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Uses the base ``checkpoint_io`` to remove the checkpoint."""
        assert self.checkpoint_io is not None
        self.checkpoint_io.remove_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Uses the base ``checkpoint_io`` to load the checkpoint."""
        assert self.checkpoint_io is not None
        return self.checkpoint_io.load_checkpoint(*args, **kwargs)
