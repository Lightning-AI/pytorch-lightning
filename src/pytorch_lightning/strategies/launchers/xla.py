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
import time
from functools import wraps
from multiprocessing.queues import SimpleQueue
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING

import torch.multiprocessing as mp
from torch.multiprocessing import ProcessContext

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers.multiprocessing import (
    _FakeQueue,
    _GlobalStateSnapshot,
    _MultiProcessingLauncher,
    _WorkerOutput,
)
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.rank_zero import rank_zero_debug

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
else:
    xm, xmp = None, None

if TYPE_CHECKING:
    from pytorch_lightning.strategies import Strategy


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

    def __init__(self, strategy: "Strategy") -> None:
        super().__init__(strategy=strategy, start_method="fork")

    @property
    def is_interactive_compatible(self) -> bool:
        return True

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
        context = mp.get_context(self._start_method)
        return_queue = context.SimpleQueue()
        _save_spawn(
            self._wrapping_function,
            args=(trainer, function, args, kwargs, return_queue),
            nprocs=len(self._strategy.parallel_devices),
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
        global_states: Optional[_GlobalStateSnapshot] = None,
    ) -> None:
        self._strategy._worker_setup(process_idx)
        results = function(*args, **kwargs)

        if trainer is not None:
            results = self._collect_rank_zero_results(trainer, results)

        if self._strategy.local_rank == 0:
            return_queue.put(move_data_to_device(results, "cpu"))

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

        # save the last weights
        weights_path = None
        if trainer.state.fn == TrainerFn.FITTING:
            weights_path = os.path.join(trainer.default_root_dir, ".temp.ckpt")
            self._strategy.checkpoint_io.save_checkpoint(state_dict, weights_path)

        # We use `local_rank` here as separate filesystems are used for each VM for TPU Pod Training
        if self._strategy.local_rank != 0:
            return None

        # adds the `callback_metrics` to the queue
        extra = _FakeQueue()
        self.add_to_queue(trainer, extra)

        return _WorkerOutput(best_model_path, weights_path, trainer.state, results, extra)


def _save_spawn(
    fn: Callable,
    args: Tuple = (),
    nprocs: Optional[int] = None,
    join: bool = True,
    daemon: bool = False,
    start_method: str = "spawn",
) -> Optional[ProcessContext]:
    """Wraps the :func:`torch_xla.distributed.xla_multiprocessing.spawn` with added teardown logic for the worker
    processes."""

    @wraps(fn)
    def wrapped(rank: int, *_args: Any) -> None:
        fn(rank, *_args)

        # Make all processes wait for each other before joining
        # https://github.com/pytorch/xla/issues/1801#issuecomment-602799542
        xm.rendezvous("end-process")

        # Ensure that the rank 0 process is the one exiting last
        # https://github.com/pytorch/xla/issues/2190#issuecomment-641665358
        if rank == 0:
            time.sleep(1)

    return xmp.spawn(wrapped, args=args, nprocs=nprocs, join=join, daemon=daemon, start_method=start_method)
