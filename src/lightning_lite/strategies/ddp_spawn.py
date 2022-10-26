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
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import Literal

from lightning_lite.accelerators.accelerator import Accelerator
from lightning_lite.plugins.collectives.torch_collective import default_pg_timeout
from lightning_lite.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_lite.plugins.io.checkpoint_io import CheckpointIO
from lightning_lite.plugins.precision import Precision
from lightning_lite.strategies.ddp import _DDPBackwardSyncControl
from lightning_lite.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning_lite.strategies.parallel import ParallelStrategy
from lightning_lite.strategies.strategy import TBroadcast
from lightning_lite.utilities.distributed import (
    _distributed_available,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning_lite.utilities.distributed import group as _group
from lightning_lite.utilities.distributed import ReduceOp
from lightning_lite.utilities.rank_zero import rank_zero_only

_DDP_FORK_ALIASES = (
    "ddp_fork",
    "ddp_fork_find_unused_parameters_false",
    "ddp_notebook",
    "ddp_notebook_find_unused_parameters_false",
)


class DDPSpawnStrategy(ParallelStrategy):
    """Spawns processes using the :func:`torch.multiprocessing.spawn` method and joins processes after training
    finishes."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["spawn", "fork", "forkserver"] = "spawn",
        **kwargs: Any,
    ):
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
        self._backward_sync_control = _DDPBackwardSyncControl()
        self._start_method = start_method
        self._ddp_kwargs = kwargs
        self._local_rank = 0

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
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def _configure_launcher(self) -> None:
        self._launcher = _MultiProcessingLauncher(self, start_method=self._start_method)

    def setup_environment(self) -> None:
        self._setup_distributed()
        super().setup_environment()

    def setup_module(self, module: Module) -> Module:
        return DistributedDataParallel(module=module, device_ids=self._determine_ddp_device_ids(), **self._ddp_kwargs)

    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    def reduce(
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
            ("ddp_spawn", "spawn"),
            ("ddp_fork", "fork"),
            ("ddp_notebook", "fork"),
        )
        for name, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `start_method` '{start_method}'",
                start_method=start_method,
            )

        entries = (
            ("ddp_spawn_find_unused_parameters_false", "spawn"),
            ("ddp_fork_find_unused_parameters_false", "fork"),
            ("ddp_notebook_find_unused_parameters_false", "fork"),
        )
        for name, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `find_unused_parameters` as False and `start_method` '{start_method}'",
                find_unused_parameters=False,
                start_method=start_method,
            )

    def _setup_distributed(self) -> None:
        self._set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(
            self.cluster_environment,
            self._process_group_backend,
            self.global_rank,
            self.world_size,
            timeout=self._timeout,
        )

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
