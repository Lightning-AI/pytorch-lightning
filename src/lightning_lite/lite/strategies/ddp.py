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
import logging
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed
from torch import Tensor
from torch.distributed.constants import default_pg_timeout
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.optimizer import Optimizer

import lightning_lite.lite as lite
from lightning_lite.lite.overrides.distributed import prepare_for_backward
from lightning_lite.lite.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_lite.lite.plugins.io.checkpoint_plugin import CheckpointIO
from lightning_lite.lite.plugins.precision import PrecisionPlugin
from lightning_lite.lite.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_lite.lite.strategies.parallel import ParallelStrategy
from lightning_lite.lite.strategies.strategy import TBroadcast
from lightning_lite.lite.utilities.distributed import (
    _get_process_group_backend_from_env,
    distributed_available,
    get_default_process_group_backend_for_device,
)
from lightning_lite.lite.utilities.distributed import group as _group
from lightning_lite.lite.utilities.distributed import init_dist_connection, ReduceOp, sync_ddp_if_available
from lightning_lite.lite.utilities.rank_zero import rank_zero_only
from lightning_lite.lite.utilities.seed import reset_seed

log = logging.getLogger(__name__)


class DDPStrategy(ParallelStrategy):
    """Strategy for multi-process single-device training on one or multiple nodes."""

    strategy_name = "ddp"

    def __init__(
        self,
        accelerator: Optional["lite.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        log.detail(f"{self.__class__.__name__}: initializing DDP plugin")
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._ddp_kwargs = kwargs
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)

    @property
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def is_distributed(self) -> bool:
        return True

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
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def _is_single_process_single_device(self) -> bool:
        return True

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    def setup_environment(self) -> None:
        self._setup_distributed()
        super().setup_environment()

    def setup_module(self, module: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        device_ids = self._determine_ddp_device_ids()
        log.detail(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
        return DistributedDataParallel(module=module, device_ids=device_ids, **self._ddp_kwargs)

    def module_to_device(self, module: Module) -> None:
        log.detail(f"{self.__class__.__name__}: moving model to device [{self.root_device}]...")
        module.to(self.root_device)

    def pre_backward(self, tensor: Tensor, module: Optional[Module]) -> None:
        """Run before precision plugin executes backward."""
        if not isinstance(module, DistributedDataParallel):
            return
        prepare_for_backward(module, tensor)

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
            tensor = sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not distributed_available():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        obj = [obj]
        if self.global_rank != src:
            obj = [None]  # type: ignore[list-item]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def teardown(
        self, modules: Iterable[Module] = (), optimizers: Iterable[Optimizer] = ()
    ) -> Tuple[Iterable[Module], Iterable[Optimizer]]:
        log.detail(f"{self.__class__.__name__}: tearing down strategy")

        modules = [(module.module if isinstance(module, DistributedDataParallel) else module) for module in modules]

        if self._layer_sync:
            modules = [self._layer_sync.revert(module) for module in modules]

        return super().teardown(modules=modules)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_find_unused_parameters_false",
            cls,
            description="DDP Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def _setup_distributed(self) -> None:
        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        # determine which process we are and world size
        self._set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return (
            self._process_group_backend
            or _get_process_group_backend_from_env()
            or get_default_process_group_backend_for_device(self.root_device)
        )

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
