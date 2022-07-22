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

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO


class AsyncCheckpointIO(_WrappingCheckpointIO):
    """``AsyncCheckpointIO`` enables saving the checkpoints asynchronously using the ``ThreadPoolExecutor``.

    .. warning::

        This is currently an experimental plugin/feature and API changes are to be expected.

    Args:
        checkpoint_io: A checkpoint IO plugin that is used as the basis for async checkpointing.
        interval: Sleep time between each queue check.
    """

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None, interval: float = 2.0) -> None:
        super().__init__(checkpoint_io)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._error = None

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Uses the ``ThreadPoolExecutor`` to save the checkpoints using the base ``checkpoint_io``."""

        def _save_checkpoint(*args, **kwargs):
            try:
                self.checkpoint_io.save_checkpoint(*args, **kwargs)
            except Exception as e:
                self._error = e

        self._executor.submit(_save_checkpoint, *args, **kwargs)

        if self._error:
            raise self._error

    def remove_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        super().remove_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return super().load_checkpoint(*args, **kwargs)

    def teardown(self) -> None:
        """This method is called to close the threads."""
        self._executor.shutdown(wait=True)
