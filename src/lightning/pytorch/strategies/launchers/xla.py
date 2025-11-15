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
import queue
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch.multiprocessing as mp
from typing_extensions import override

from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.pytorch.strategies.launchers.multiprocessing import (
    _GlobalStateSnapshot,
    _MultiProcessingLauncher,
    _WorkerOutput,
)

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
        super().__init__(strategy)
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.strategies.xla.launcher import _XLALauncherTrainer as EnterpriseXLALauncher

        self.xla_launcher_impl = EnterpriseXLALauncher(strategy)

    @property
    @override
    def is_interactive_compatible(self) -> bool:
        return self.xla_launcher_impl.is_interactive_compatible

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
        return self.xla_launcher_impl.launch(function, *args, trainer=trainer, **kwargs)

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
        return self.xla_launcher_impl._wrapping_function(
            process_idx, trainer, function, args, kwargs, return_queue, global_states
        )

    @override
    def _collect_rank_zero_results(self, trainer: "pl.Trainer", results: Any) -> Optional["_WorkerOutput"]:
        return self.xla_launcher_impl._collect_rank_zero_results(trainer, results)
