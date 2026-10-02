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

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from typing_extensions import override

from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO

if TYPE_CHECKING:
    from lightning.fabric.plugins import CheckpointIO


class AsyncCheckpointIO(_WrappingCheckpointIO):
    """``AsyncCheckpointIO`` enables saving checkpoints asynchronously in a single thread.

    While the executor is active, save and removal requests are executed in order. Removals before the first save or
    after :meth:`teardown` run synchronously. Call :meth:`teardown` to wait for pending operations to finish.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        checkpoint_io: A checkpoint IO plugin that is used as the basis for async checkpointing.

    """

    _executor: Optional[ThreadPoolExecutor]
    _error: Optional[BaseException]

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None) -> None:
        super().__init__(checkpoint_io)
        self._executor = None
        self._error = None

    # CheckpointIO doesn't have a setup method so we have to do something like.
    def _ensure_setup(self) -> None:
        """Ensures that the executor is setup.

        We can't do setup in __init__ because if train or validate is called more than once, the teardown method deletes
        the executor.

        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)

    @override
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Uses the ``ThreadPoolExecutor`` to save the checkpoints using the base ``checkpoint_io``."""

        # rebuild args/kwargs with a cloned checkpoint (supports positional or kw form)
        if "checkpoint" in kwargs:
            kwargs = {**kwargs, "checkpoint": apply_to_collection(kwargs["checkpoint"], torch.Tensor, _clone_tensor)}
        elif len(args) >= 1:
            args = (apply_to_collection(args[0], torch.Tensor, _clone_tensor), *args[1:])

        self._submit(super().save_checkpoint, *args, **kwargs)

    @override
    def remove_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Queues removal after earlier checkpoint operations to avoid racing with pending saves."""
        if self._executor is None:
            # Callback teardown can remove checkpoints after the strategy has already shut down the executor.
            super().remove_checkpoint(*args, **kwargs)
            return
        self._submit(super().remove_checkpoint, *args, **kwargs)

    def _submit(self, operation: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        self._ensure_setup()

        def _run_operation() -> None:
            try:
                operation(*args, **kwargs)
            except BaseException as ex:
                self._error = ex

        assert self._executor is not None
        self._executor.submit(_run_operation)

        # if an error was raised between the previous checkpoint operation and now,
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


# snapshot the checkpoint payload on the caller thread to avoid races with parameter mutation
def _clone_tensor(t: torch.Tensor) -> torch.Tensor:
    """Clones a tensor on the caller thread."""
    # detach to avoid autograd history and clone to take a point-in-time copy
    return t.detach().clone()
