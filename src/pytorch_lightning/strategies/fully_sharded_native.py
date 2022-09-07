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
import contextlib
import logging
from typing import Any, Dict, Generator, List, Optional, Union

import torch
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.distributed import (
    _get_process_group_backend_from_env,
    get_default_process_group_backend_for_device,
)
from pytorch_lightning.utilities.distributed import group as _group
from pytorch_lightning.utilities.distributed import init_dist_connection, ReduceOp, sync_ddp_if_available
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from pytorch_lightning.utilities.optimizer import optimizers_to_device
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import ProcessGroup

_distributed_available = torch.distributed.is_available()
_fsdp_available = _TORCH_GREATER_EQUAL_1_12 and _distributed_available
if _fsdp_available:
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        BackwardPrefetch,
        CPUOffload,
        FullyShardedDataParallel,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import enable_wrap
else:
    MixedPrecision = None
    BackwardPrefetch = None  # type: ignore[misc,assignment]
    CPUOffload = None  # type: ignore[misc,assignment]

if _distributed_available:
    from torch.distributed.distributed_c10d import _get_default_group

log = logging.getLogger(__name__)


class DDPFullyShardedNativeStrategy(ParallelStrategy):

    strategy_name = "fsdp_native"
    _registered_strategies: List[str] = []

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = None,
        cpu_offload: Optional[CPUOffload] = None,
        backward_prefetch: Optional[BackwardPrefetch] = None,
        mixed_precision: Optional[MixedPrecision] = None,
        **kwargs: Any,
    ) -> None:
        r"""Strategy for Fully Sharded Data Parallel provided by torch.Distributed.

        Fully Sharded Training shards the entire model across all available GPUs, allowing you to scale model
        size, whilst using efficient communication to reduce overhead. In practice, this means we can remain
        at parity with PyTorch DDP, whilst scaling our model sizes dramatically. The technique is similar
        to ZeRO-Stage 3.
        `For more information: https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/`.

        .. warning:: ``DDPFullyShardedNativeStrategy`` is in beta and subject to change. The interface can
        bring breaking changes and new features with the next release of PyTorch.

        Defaults have been set and options have been exposed, but may require configuration
        based on your level of memory/speed efficiency. We suggest having a look at this tutorial for
        more information.
        `https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html`

        Arguments:
            cpu_offload:
                CPU offloading config. Currently, only parameter and gradient CPU
                offload is supported. It can be enabled via passing in
                ``cpu_offload=CPUOffload(offload_params=True)``. Note that this
                currently implicitly enables gradient offloading to CPU in order for
                params and grads to be on same device to work with optimizer. This
                API is subject to change. Default is ``None`` in which case there
                will be no offloading.
            backward_prefetch:
                This is an experimental feature that is subject to change in the
                the near future. It allows users to enable two different backward_prefetch
                algorithms to help backward communication and computation overlapping.
                Pros and cons of each algorithm is explained in the class ``BackwardPrefetch``.
            mixed_precision:
                Mixed Precision config. By default, Lightning will enable FP16 if ``precision=16`
                or BF16 if ``precision=bf16`` unless a config is passed in.
                This is only available in PyTorch 1.12 and later.
            \**kwargs: Passed to the FSDP Context manager which will configure the FSDP class when wrapping modules.
        """
        if not _TORCH_GREATER_EQUAL_1_12:
            raise MisconfigurationException(
                "`DDPFullyShardedNativeStrategy` is supported from PyTorch v1.12.0 onwards."
            )

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self._process_group = None
        self.num_nodes = 1
        self._process_group_backend = process_group_backend
        self.cpu_offload = cpu_offload
        self.backward_prefetch = backward_prefetch
        self.mixed_precision = mixed_precision
        self._rank_0_will_call_children_scripts: bool = False
        self.kwargs = kwargs

    @property
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def process_group(self) -> Optional[ProcessGroup]:
        if self._process_group is None:
            # The strategy should have already initilized process group in setup_environment()
            self._process_group = _get_default_group()
        return self._process_group

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    def mixed_precision_config(self) -> Optional[MixedPrecision]:
        if self.mixed_precision:
            return self.mixed_precision
        plugin = self.precision_plugin
        if isinstance(plugin, FullyShardedNativeNativeMixedPrecisionPlugin):
            return plugin.mixed_precision_config

    @property
    def distributed_sampler_kwargs(self) -> Dict:
        return dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)

    def setup_environment(self) -> None:
        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        init_dist_connection(self.cluster_environment, self._process_group_backend)
        super().setup_environment()

    def _get_process_group_backend(self) -> str:
        return (
            self._process_group_backend
            or _get_process_group_backend_from_env()
            or get_default_process_group_backend_for_device(self.root_device)
        )

    def set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)
            self._rank_0_will_call_children_scripts = True

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        # share ddp pids to all processes
        self._rank_0_will_call_children_scripts = self.broadcast(self._rank_0_will_call_children_scripts)

        if trainer.state.fn == TrainerFn.FITTING and self._layer_sync:
            assert self.model is not None
            self.model = self._layer_sync.apply(self.model)

        # we set the device so that optimizers can be created with distributed comms.
        assert self.lightning_module is not None
        self.lightning_module._device = self.root_device

        self.barrier()
        self.setup_optimizers(trainer)
        optimizers_to_device(self.optimizers, self.root_device)
        self.setup_precision_plugin()

    def model_to_device(self) -> None:
        pass

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator:
        log.detail(f"{self.__class__.__name__}: entered model_sharded_context.")
        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            process_group=self.process_group,
            cpu_offload=self.cpu_offload,
            backward_prefetch=self.backward_prefetch,
            mixed_precision=self.mixed_precision_config,
            device_id=self.root_device.index,
            **self.kwargs,
        ):
            yield

    def barrier(self, name: Optional[str] = None) -> None:
        if not _distributed_available:
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        obj = [obj]
        if self.global_rank != src:
            obj = [None]  # type: ignore
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
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

    def _determine_device_ids(self) -> List[int]:
        return [self.root_device.index]

    def teardown(self) -> None:
        rank_zero_info(f"{self.__class__.__name__}: tearing down strategy...")

        pl_module = self.lightning_module
        if (
            pl_module is not None
            # `self.lightning_module._trainer` can be None if teardown gets called on an exception before
            # the trainer gets set on the LightningModule
            and pl_module._trainer is not None
            and pl_module._trainer.state.fn == TrainerFn.FITTING
            and self._layer_sync
        ):
            assert self.model is not None
            self.model = self._layer_sync.revert(self.model)

        assert self.cluster_environment is not None
        assert self.accelerator is not None
        self.cluster_environment.teardown()
        self.precision_plugin.teardown()
        self.accelerator.teardown()

    @classmethod
    def get_registered_strategies(cls) -> List[str]:
        return cls._registered_strategies

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        if _fsdp_available:
            strategy_registry.register(
                "fsdp_native",
                cls,
                description="Fully Sharded Data Parallel training from torch.distributed.",
            )
            cls._registered_strategies.append("fsdp_native")

            strategy_registry.register(
                "fsdp_native_full_shard_offload",
                cls,
                description="Native FSDP with Full Sharding and CPU Offloading",
                cpu_offload=CPUOffload(offload_params=True),
            )
            cls._registered_strategies.append("fsdp_native_full_shard_offload")
