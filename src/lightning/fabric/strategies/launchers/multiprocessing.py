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
import itertools
import os
from dataclasses import dataclass
from multiprocessing.queues import SimpleQueue
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
from lightning_utilities import apply_to_collection
from torch.nn import Module
from typing_extensions import override

from lightning.fabric.accelerators.cpu import CPUAccelerator
from lightning.fabric.strategies.launchers.launcher import _Launcher
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.fabric.utilities.distributed import _set_num_threads_if_needed
from lightning.fabric.utilities.imports import _IS_INTERACTIVE
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states

if TYPE_CHECKING:
    from lightning.fabric.strategies import ParallelStrategy


class _MultiProcessingLauncher(_Launcher):
    r"""Launches processes that run a given function in parallel, and joins them all at the end.

    The main process in which this launcher is invoked creates N so-called worker processes (using
    :func:`torch.multiprocessing.start_processes`) that run the given function.
    Worker processes have a rank that ranges from 0 to N - 1.

    Note:
        - This launcher requires all objects to be pickleable.
        - It is important that the entry point to the program/script is guarded by ``if __name__ == "__main__"``.
        - With start method 'fork' the user must ensure that no CUDA context gets created in the main process before
          the launcher is invoked. E.g., one should avoid creating cuda tensors or calling ``torch.cuda.*`` functions
          before calling ``Trainer.fit``.

    Args:
        strategy: A reference to the strategy that is used together with this launcher.
        start_method: The method how to start the processes.
            - 'spawn': The default start method. Requires all objects to be pickleable.
            - 'fork': Preferable for IPython/Jupyter environments where 'spawn' is not available. Not available on
              the Windows platform for example.
            - 'forkserver': Alternative implementation to 'fork'.

    """

    def __init__(
        self,
        strategy: "ParallelStrategy",
        start_method: Literal["spawn", "fork", "forkserver"] = "spawn",
    ) -> None:
        self._strategy = strategy
        self._start_method = start_method
        if start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"The start method '{self._start_method}' is not available on this platform. Available methods are:"
                f" {', '.join(mp.get_all_start_methods())}"
            )

    @property
    @override
    def is_interactive_compatible(self) -> bool:
        # The start method 'spawn' is not supported in interactive environments
        # The start method 'fork' is the only one supported in Jupyter environments, with constraints around CUDA
        # initialization. For more context, see https://github.com/Lightning-AI/pytorch-lightning/issues/7550
        return self._start_method == "fork"

    @override
    def launch(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            **kwargs: Optional keyword arguments to be passed to the given function.

        """
        if self._start_method in ("fork", "forkserver"):
            _check_bad_cuda_fork()
        if self._start_method == "spawn":
            _check_missing_main_guard()

        # The default cluster environment in Lightning chooses a random free port number
        # This needs to be done in the main process here before starting processes to ensure each rank will connect
        # through the same port
        assert self._strategy.cluster_environment is not None
        os.environ["MASTER_PORT"] = str(self._strategy.cluster_environment.main_port)

        context = mp.get_context(self._start_method)
        return_queue = context.SimpleQueue()

        if self._start_method == "spawn":
            global_states = _GlobalStateSnapshot.capture()
            process_args = [function, args, kwargs, return_queue, global_states]
        else:
            process_args = [function, args, kwargs, return_queue]

        mp.start_processes(
            self._wrapping_function,
            args=process_args,
            nprocs=self._strategy.num_processes,
            start_method=self._start_method,
        )
        return return_queue.get()

    def _wrapping_function(
        self,
        process_idx: int,
        function: Callable,
        args: Any,
        kwargs: Any,
        return_queue: SimpleQueue,
        global_states: Optional["_GlobalStateSnapshot"] = None,
    ) -> None:
        if global_states:
            global_states.restore()
        if self._start_method == "spawn" and isinstance(self._strategy.accelerator, CPUAccelerator):
            args, kwargs = _disable_module_memory_sharing((args, kwargs))

        _set_num_threads_if_needed(num_processes=self._strategy.num_processes)

        os.environ["LOCAL_RANK"] = str(process_idx)
        results = function(*args, **kwargs)

        if process_idx == 0:
            return_queue.put(move_data_to_device(results, "cpu"))


@dataclass
class _GlobalStateSnapshot:
    """Captures a hand-selected set of (global) variables in modules and provides a way to restore them.

    It facilitates and encapsulates the transfer of globals like PyTorch's deterministic flags or random generator state
    across process boundaries when launching processes with :func:`torch.multiprocessing.spawn`.

    Example:

        .. code-block:: python

            # in main process
            snapshot = _GlobalStateSnapshot.capture()

            # in worker process
            snapshot.restore()

    """

    use_deterministic_algorithms: bool
    use_deterministic_algorithms_warn_only: bool
    cudnn_benchmark: bool
    rng_states: dict[str, Any]

    @classmethod
    def capture(cls) -> "_GlobalStateSnapshot":
        """Capture a few global states from torch, numpy, etc., that we want to restore in a spawned worker process."""
        return cls(
            use_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            use_deterministic_algorithms_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            rng_states=_collect_rng_states(),
        )

    def restore(self) -> None:
        """Restores all globals to the values captured in the :meth:`capture` method."""
        torch.use_deterministic_algorithms(
            self.use_deterministic_algorithms, warn_only=self.use_deterministic_algorithms_warn_only
        )
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        _set_rng_states(self.rng_states)


def _check_bad_cuda_fork() -> None:
    """Checks whether it is safe to fork and initialize CUDA in the new processes, and raises an exception if not.

    The error message replaces PyTorch's 'Cannot re-initialize CUDA in forked subprocess' with helpful advice for
    Lightning users.

    """
    if not torch.cuda.is_initialized():
        return

    message = (
        "Lightning can't create new processes if CUDA is already initialized. Did you manually call"
        " `torch.cuda.*` functions, have moved the model to the device, or allocated memory on the GPU any"
        " other way? Please remove any such calls, or change the selected strategy."
    )
    if _IS_INTERACTIVE:
        message += " You will have to restart the Python kernel."
    raise RuntimeError(message)


def _disable_module_memory_sharing(data: Any) -> Any:
    """Disables memory sharing on parameters and buffers of `nn.Module`s contained in the given collection.

    Note: This is only required when running on CPU.

    """
    # PyTorch enables memory sharing automatically on all tensors that are passed through `mp.spawn`.
    # For model weights and buffers, this is undesired and can lead to race conditions between processes.
    # Hence, we copy the tensors in the entire module to ensure it doesn't share memory with other processes.

    @torch.no_grad()
    def unshare(module: Module) -> Module:
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            tensor.data = tensor.data.clone()
        return module

    return apply_to_collection(data, function=unshare, dtype=Module)


def _check_missing_main_guard() -> None:
    """Raises an exception if the ``__name__ == "__main__"`` guard is missing."""
    if not getattr(mp.current_process(), "_inheriting", False):
        return
    message = dedent(
        """
        Launching multiple processes with the 'spawn' start method requires that your script guards the main
        function with an `if __name__ == \"__main__\"` clause. For example:

        def main():
            # Put your code here
            ...

        if __name__ == "__main__":
            main()

        Alternatively, you can run with `strategy="ddp"` to avoid this error.
        """
    )
    raise RuntimeError(message)
