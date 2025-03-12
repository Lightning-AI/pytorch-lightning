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
import queue
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch.multiprocessing as mp
from typing_extensions import override

from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.strategies.launchers.xla import _rank_teardown
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.strategies.launchers.multiprocessing import (
    _GlobalStateSnapshot,
    _MultiProcessingLauncher,
    _WorkerOutput,
)
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.rank_zero import rank_zero_debug

if TYPE_CHECKING:
    import lightning.pytorch as pl


class _XLALauncher(_MultiProcessingLauncher):
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

    def __init__(self, strategy: "pl.strategies.XLAStrategy") -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(strategy=strategy, start_method="fork")

    @property
    @override
    def is_interactive_compatible(self) -> bool:
        return True

    @override
    def launch(self, function: Callable, *args: Any, trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~lightning.pytorch.trainer.trainer.Trainer` for which
                a selected set of attributes get restored in the main process after processes join.
            **kwargs: Optional keyword arguments to be passed to the given function.

        """
        if self._already_fit and trainer is not None and trainer.state.fn == TrainerFn.FITTING:
            # resolving https://github.com/Lightning-AI/pytorch-lightning/issues/18775 will lift this restriction
            raise NotImplementedError(
                "Calling `trainer.fit()` twice on the same Trainer instance using a spawn-based strategy is not"
                " supported. You can work around this by creating a new Trainer instance and passing the"
                " `fit(ckpt_path=...)` argument."
            )

        # pjrt requires that the queue is serializable
        return_queue = mp.Manager().Queue()

        import torch_xla.distributed.xla_multiprocessing as xmp

        spawn_kwargs = {}
        nprocs = self._strategy.num_processes
        if nprocs == 1:
            # avoid warning: "Unsupported nprocs". If it's 1, it will call the launched function directly.
            # otherwise it will use all devices
            spawn_kwargs["nprocs"] = nprocs

        process_context = xmp.spawn(
            self._wrapping_function,
            args=(trainer, function, args, kwargs, return_queue),
            start_method=self._start_method,
            join=False,  # we will join ourselves to get the process references
            **spawn_kwargs,
        )
        # xla will not actually create processes if only 1 device
        if process_context is not None:
            self.procs = process_context.processes
            while not process_context.join():
                pass

        worker_output = return_queue.get()
        if trainer is None:
            return worker_output

        self._already_fit |= trainer.state.fn == TrainerFn.FITTING
        self._recover_results_in_main_process(worker_output, trainer)
        return worker_output.trainer_results

    @override
    def _wrapping_function(
        self,
        # XLA's multiprocessing returns the global index, not the local index as torch's multiprocessing
        # https://github.com/pytorch/xla/blob/v1.13.0/torch_xla/distributed/xla_multiprocessing.py#L321
        process_idx: int,
        trainer: Optional["pl.Trainer"],
        function: Callable,
        args: Any,
        kwargs: Any,
        return_queue: Union[mp.SimpleQueue, queue.Queue],
        global_states: Optional[_GlobalStateSnapshot] = None,
    ) -> None:
        import torch_xla.core.xla_model as xm

        if len(xm.get_xla_supported_devices()) > 1:
            # `get_xla_supported_devices` in the spawned process returns the logical devices (2 for v2/v3 and 1 for v4)
            # so when there's more than one (multithreading), objects need to be deep-copied
            import copy

            trainer, function, args, kwargs = copy.deepcopy((trainer, function, args, kwargs))

        results = function(*args, **kwargs)

        if trainer is not None:
            results = self._collect_rank_zero_results(trainer, results)

        if self._strategy.local_rank == 0:
            return_queue.put(move_data_to_device(results, "cpu"))

        _rank_teardown(self._strategy.local_rank)

    @override
    def _collect_rank_zero_results(self, trainer: "pl.Trainer", results: Any) -> Optional["_WorkerOutput"]:
        rank_zero_debug("Collecting results from rank 0 process.")
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = (
            checkpoint_callback.best_model_path
            if checkpoint_callback and hasattr(checkpoint_callback, "best_model_path")
            else None
        )

        # save the last weights
        weights_path = None
        if trainer.state.fn == TrainerFn.FITTING:
            # requires to compute the state_dict on all processes in case Metrics are present
            state_dict = self._strategy.lightning_module_state_dict()
            weights_path = os.path.join(trainer.default_root_dir, ".temp.ckpt")
            self._strategy.checkpoint_io.save_checkpoint(state_dict, weights_path)

        # We use `local_rank` here as separate filesystems are used for each VM for TPU Pod Training
        if self._strategy.local_rank != 0:
            return None

        # add extra result data from trainer to send to main process
        extra = self.get_extra_results(trainer)

        return _WorkerOutput(best_model_path, weights_path, trainer.state, results, extra)
