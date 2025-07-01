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

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from typing_extensions import override

from lightning.fabric.plugins import CheckpointIO
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO


class AsyncCheckpointIO(_WrappingCheckpointIO):
    """``AsyncCheckpointIO`` enables saving the checkpoints asynchronously in a thread.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        checkpoint_io: A checkpoint IO plugin that is used as the basis for async checkpointing.

    """

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None) -> None:
        super().__init__(checkpoint_io)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._error: Optional[BaseException] = None

    # CheckpointIO doesn't have a setup method so we have to do something like.
    # We can't do setup in __init__ because if train or validate is called more than once the
    # teardown method deletes the executor.
    def _ensure_setup(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._error: Optional[BaseException] = None

    @override
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Uses the ``ThreadPoolExecutor`` to save the checkpoints using the base ``checkpoint_io``."""

        self._ensure_setup()

        def _save_checkpoint(*args: Any, **kwargs: Any) -> None:
            try:
                assert self.checkpoint_io is not None
                self.checkpoint_io.save_checkpoint(*args, **kwargs)
            except BaseException as ex:
                self._error = ex

        self._executor.submit(_save_checkpoint, *args, **kwargs)

        # if an error was raised between the previous time `save_checkpoint`` was called and now,
        # because `executor.submit` is not blocking
        if self._error:
            raise self._error

    @override
    def teardown(self) -> None:
        """This method is called to close the threads."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

            # if an error was raised anytime in any of the `executor.submit` calls
            if self._error:
                raise self._error
