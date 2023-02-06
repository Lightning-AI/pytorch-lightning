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
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, Generator, List, Optional, Union

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import Literal

from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning_fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.strategy import _BackwardSyncControl, TBroadcast
from lightning_fabric.utilities.distributed import (
    _distributed_available,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.distributed import ReduceOp
from lightning_fabric.utilities.rank_zero import rank_zero_only

_DDP_FORK_ALIASES = (
    "ddp_fork",
    "ddp_notebook",
)


class DDPStrategy(ParallelStrategy):
    """Strategy for multi-process single-device training on one or multiple nodes."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._start_method = start_method
        self._backward_sync_control = _DDPBackwardSyncControl()
        self._ddp_kwargs = kwargs

    @property
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        return dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if self._start_method == "popen":
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)
        else:
            self._launcher = _MultiProcessingLauncher(self, start_method=self._start_method)

    def setup_environment(self) -> None:
        self._setup_distributed()
        super().setup_environment()

    def setup_module(self, module: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        return DistributedDataParallel(module=module, device_ids=self._determine_ddp_device_ids(), **self._ddp_kwargs)

    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if isinstance(tensor, Tensor):
            tensor = _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_available():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_available():
            return obj
        obj = [obj]
        if self.global_rank != src:
            obj = [None]  # type: ignore[list-item]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        entries = (
            ("ddp", "popen"),
            ("ddp_spawn", "spawn"),
            ("ddp_fork", "fork"),
            ("ddp_notebook", "fork"),
        )
        for name, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `start_method={start_method!r}`",
                start_method=start_method,
            )

    def _setup_distributed(self) -> None:
        self._set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

    def _determine_ddp_device_ids(self) -> Optional[List[int]]:
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]


class _DDPBackwardSyncControl(_BackwardSyncControl):
    @contextmanager
    def no_backward_sync(self, module: Module) -> Generator:
        """Blocks gradient synchronization inside the
        :class:`~torch.nn.parallel.distributed.DistributedDataParallel` wrapper."""
        if not isinstance(module, DistributedDataParallel):
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{self.__class__.__name__}.no_backward_sync` is wrapped in `DistributedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        with module.no_sync():  # type: ignore[operator]
            yield
