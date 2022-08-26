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
from collections import UserList
from dataclasses import dataclass
from multiprocessing.queues import SimpleQueue
from typing import Any, Callable, Dict, NamedTuple, Optional

import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
from torch import Tensor
from typing_extensions import Literal

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers.base import _Launcher
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.states import TrainerFn, TrainerState
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_11
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from pytorch_lightning.utilities.seed import _collect_rng_states, _set_rng_states
from pytorch_lightning.utilities.types import _PATH


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
            - 'fork': Preferrable for IPython/Jupyter environments where 'spawn' is not available. Not available on
              the Windows platform for example.
            - 'forkserver': Alternative implementation to 'fork'.
    """

    def __init__(self, strategy: Strategy, start_method: Literal["spawn", "fork", "forkserver"] = "spawn") -> None:
        self._strategy = strategy
        self._start_method = start_method
        if start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"The start method '{self._start_method}' is not available on this platform. Available methods are:"
                f" {', '.join(mp.get_all_start_methods())}"
            )
        if start_method in ("fork", "forkserver") and _is_forking_disabled():
            raise ValueError(
                "Forking is disabled in this environment by `PL_DISABLE_FORKING=1`. Choose a different start method."
            )

    @property
    def is_interactive_compatible(self) -> bool:
        # The start method 'spawn' is not supported in interactive environments
        # The start method 'fork' is the only one supported in Jupyter environments, with constraints around CUDA
        # initialization. For more context, see https://github.com/Lightning-AI/lightning/issues/7550
        return self._start_method == "fork"

    def launch(self, function: Callable, *args: Any, trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~pytorch_lightning.trainer.trainer.Trainer` for which
                a selected set of attributes get restored in the main process after processes join.
            **kwargs: Optional keyword arguments to be passed to the given function.
        """
        # The default cluster environment in Lightning chooses a random free port number
        # This needs to be done in the main process here before starting processes to ensure each rank will connect
        # through the same port
        os.environ["MASTER_PORT"] = str(self._strategy.cluster_environment.main_port)
        context = mp.get_context(self._start_method)
        return_queue = context.SimpleQueue()

        if self._start_method == "spawn":
            global_states = _GlobalStateSnapshot.capture()
            process_args = [trainer, function, args, kwargs, return_queue, global_states]
        else:
            process_args = [trainer, function, args, kwargs, return_queue]

        mp.start_processes(
            self._wrapping_function,
            args=process_args,
            nprocs=self._strategy.num_processes,
            start_method=self._start_method,
        )
        worker_output = return_queue.get()
        if trainer is None:
            return worker_output

        self._recover_results_in_main_process(worker_output, trainer)
        return worker_output.trainer_results

    def _wrapping_function(
        self,
        process_idx: int,
        trainer: Optional["pl.Trainer"],
        function: Callable,
        args: Any,
        kwargs: Any,
        return_queue: SimpleQueue,
        global_states: Optional["_GlobalStateSnapshot"] = None,
    ) -> None:
        if global_states:
            global_states.restore()
        self._strategy._worker_setup(process_idx)
        results = function(*args, **kwargs)

        if trainer is not None:
            results = self._collect_rank_zero_results(trainer, results)

        if self._strategy.local_rank == 0:
            return_queue.put(move_data_to_device(results, "cpu"))

    def _recover_results_in_main_process(self, worker_output: "_WorkerOutput", trainer: "pl.Trainer") -> None:
        # transfer back the best path to the trainer
        if trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, "best_model_path"):
            trainer.checkpoint_callback.best_model_path = str(worker_output.best_model_path)

        # TODO: pass also best score
        # load last weights
        if worker_output.weights_path is not None:
            ckpt = self._strategy.checkpoint_io.load_checkpoint(worker_output.weights_path)
            trainer.lightning_module.load_state_dict(ckpt)  # type: ignore[arg-type]
            self._strategy.checkpoint_io.remove_checkpoint(worker_output.weights_path)

        trainer.state = worker_output.trainer_state

        # get the `callback_metrics` and set it to the trainer
        self.get_from_queue(trainer, worker_output.extra)

    def _collect_rank_zero_results(self, trainer: "pl.Trainer", results: Any) -> Optional["_WorkerOutput"]:
        rank_zero_debug("Collecting results from rank 0 process.")
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = (
            checkpoint_callback.best_model_path
            if checkpoint_callback and hasattr(checkpoint_callback, "best_model_path")
            else None
        )

        # requires to compute the state_dict on all processes in case Metrics are present
        state_dict = trainer.lightning_module.state_dict()

        if self._strategy.global_rank != 0:
            return None

        # save the last weights
        weights_path = None
        if trainer.state.fn == TrainerFn.FITTING:
            weights_path = os.path.join(trainer.default_root_dir, ".temp.ckpt")
            self._strategy.checkpoint_io.save_checkpoint(state_dict, weights_path)

        # adds the `callback_metrics` to the queue
        extra = _FakeQueue()
        self.add_to_queue(trainer, extra)

        return _WorkerOutput(best_model_path, weights_path, trainer.state, results, extra)

    def add_to_queue(self, trainer: "pl.Trainer", queue: "_FakeQueue") -> None:
        """Appends the :attr:`trainer.callback_metrics` dictionary to the given queue. To avoid issues with memory
        sharing, we cast the data to numpy.

        Args:
            trainer: reference to the Trainer.
            queue: the instance of the queue to append the data.
        """
        callback_metrics: dict = apply_to_collection(
            trainer.callback_metrics, Tensor, lambda x: x.cpu().numpy()
        )  # send as numpy to avoid issues with memory sharing
        queue.put(callback_metrics)

    def get_from_queue(self, trainer: "pl.Trainer", queue: "_FakeQueue") -> None:
        """Retrieve the :attr:`trainer.callback_metrics` dictionary from the given queue. To preserve consistency,
        we cast back the data to ``torch.Tensor``.

        Args:
            trainer: reference to the Trainer.
            queue: the instance of the queue from where to get the data.
        """
        # NOTE: `add_to_queue` needs to be called before
        callback_metrics: dict = queue.get()
        trainer.callback_metrics.update(apply_to_collection(callback_metrics, np.ndarray, lambda x: torch.tensor(x)))


class _FakeQueue(UserList):
    """Simulates a :class:`torch.multiprocessing.queue.SimpleQueue` interface using the Python list."""

    def get(self) -> Any:
        return self.pop(0)

    def put(self, item: Any) -> None:
        self.append(item)

    def empty(self) -> bool:
        return len(self) == 0


class _WorkerOutput(NamedTuple):
    best_model_path: Optional[_PATH]
    weights_path: Optional[_PATH]
    trainer_state: TrainerState
    trainer_results: Any
    extra: _FakeQueue


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
    rng_states: Dict[str, Any]

    @classmethod
    def capture(cls) -> "_GlobalStateSnapshot":
        """Capture a few global states from torch, numpy, etc., that we want to restore in a spawned worker
        process."""
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled() if _TORCH_GREATER_EQUAL_1_11 else False
        return cls(
            use_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            use_deterministic_algorithms_warn_only=warn_only,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            rng_states=_collect_rng_states(),
        )

    def restore(self) -> None:
        """Restores all globals to the values captured in the :meth:`capture` method."""
        if _TORCH_GREATER_EQUAL_1_11:
            torch.use_deterministic_algorithms(
                self.use_deterministic_algorithms, warn_only=self.use_deterministic_algorithms_warn_only
            )
        else:
            torch.use_deterministic_algorithms(self.use_deterministic_algorithms)
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        _set_rng_states(self.rng_states)


def _is_forking_disabled() -> bool:
    """Returns whether forking is disabled through the environment variable ``PL_DISABLE_FORK``."""
    return bool(int(os.environ.get("PL_DISABLE_FORK", "0")))
