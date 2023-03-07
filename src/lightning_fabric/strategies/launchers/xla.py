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
import time
from multiprocessing.queues import SimpleQueue
from typing import Any, Callable, Optional, TYPE_CHECKING

from torch.multiprocessing import get_context

from lightning_fabric.accelerators.tpu import _XLA_AVAILABLE
from lightning_fabric.strategies.launchers.base import _Launcher
from lightning_fabric.strategies.launchers.multiprocessing import _GlobalStateSnapshot
from lightning_fabric.utilities.apply_func import move_data_to_device

if TYPE_CHECKING:
    from lightning_fabric.strategies import XLAStrategy


class _XLALauncher(_Launcher):
    r"""Launches processes that run a given function in parallel on XLA supported hardware, and joins them all at the
    end.

    The main process in which this launcher is invoked creates N so-called worker processes (using the
    `torch_xla` :func:`xmp.spawn`) that run the given function.
    Worker processes have a rank that ranges from 0 to N - 1.

    Note:
        - This launcher requires all objects to be pickleable.
        - It is important that the entry point to the program/script is guarded by ``if __name__ == "__main__"``.

    Args:
        strategy: A reference to the strategy that is used together with this launcher
    """

    def __init__(self, strategy: "XLAStrategy") -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        self._strategy = strategy
        self._start_method = "fork"

    @property
    def is_interactive_compatible(self) -> bool:
        return True

    def launch(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            **kwargs: Optional keyword arguments to be passed to the given function.
        """
        context = get_context(self._start_method)
        return_queue = context.SimpleQueue()
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(
            self._wrapping_function,
            args=(function, args, kwargs, return_queue),
            nprocs=self._strategy.num_processes,
            start_method=self._start_method,
        )
        return return_queue.get()

    def _wrapping_function(
        self,
        # XLA's multiprocessing returns the global index, not the local index as torch's multiprocessing
        # https://github.com/pytorch/xla/blob/v1.13.0/torch_xla/distributed/xla_multiprocessing.py#L321
        process_idx: int,
        function: Callable,
        args: Any,
        kwargs: Any,
        return_queue: SimpleQueue,
        global_states: Optional[_GlobalStateSnapshot] = None,
    ) -> None:
        results = function(*args, **kwargs)

        if self._strategy.local_rank == 0:
            return_queue.put(move_data_to_device(results, "cpu"))

        _rank_teardown(self._strategy.local_rank)


def _rank_teardown(rank: int) -> None:
    import torch_xla.core.xla_model as xm

    # Make all processes wait for each other before joining
    # https://github.com/pytorch/xla/issues/1801#issuecomment-602799542
    xm.rendezvous("end-process")
    # Ensure that the rank 0 process is the one exiting last
    # https://github.com/pytorch/xla/issues/2190#issuecomment-641665358
    if rank == 0:
        time.sleep(1)
