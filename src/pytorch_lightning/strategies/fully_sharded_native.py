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
import contextlib
import logging
from typing import Any, Dict, Generator, List, Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Module

import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.strategies.fsdp import (
    _init_cpu_offload,
    _optimizer_has_flat_params,
    _setup_activation_checkpointing,
)
from lightning_fabric.utilities.distributed import (
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import ProcessGroup, ReduceOp
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_13
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT

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
    FullyShardedDataParallel = None  # type: ignore[misc,assignment]
    MixedPrecision = None  # type: ignore[misc,assignment]
    BackwardPrefetch = None  # type: ignore[misc,assignment]
    CPUOffload = None  # type: ignore[misc,assignment]

if _distributed_available:
    from torch.distributed.distributed_c10d import _get_default_group

log = logging.getLogger(__name__)


class DDPFullyShardedNativeStrategy(ParallelStrategy):
    r"""Strategy for Fully Sharded Data Parallel provided by torch.distributed.

    .. warning:: ``DDPFullyShardedNativeStrategy`` is in BETA and subject to change. The interface can
        bring breaking changes and new features with the next release of PyTorch.

    Fully Sharded Training shards the entire model across all available GPUs, allowing you to scale model
    size, whilst using efficient communication to reduce overhead. In practice, this means we can remain
    at parity with PyTorch DDP, whilst scaling our model sizes dramatically. The technique is similar
    to ZeRO-Stage 3.

    For more information `check out <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api>`__.

    Defaults have been set and options have been exposed, but may require configuration
    based on your level of memory/speed efficiency. We suggest having a look at
    `this tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__ for more information.

    Arguments:
        cpu_offload: Enable offloading parameters and gradients to CPU to save GPU memory at the cost of speed.
            You can also pass a config: ``cpu_offload=CPUOffload(offload_params=True)``. Note that this currently
            implicitly enables gradient offloading to CPU in order for parameters and gradients to be on same device
            to work with the optimizer. This API is subject to change. Default: no offoading
        backward_prefetch:
            This is an experimental feature that is subject to change in the
            the near future. It allows users to enable two different backward_prefetch
            algorithms to help backward communication and computation overlapping.
            The pros and cons of each algorithm is explained in the class ``BackwardPrefetch``.
        mixed_precision:
            Mixed Precision config. By default, Lightning will enable FP16 if ``precision=16``
            or BF16 if ``precision=bf16`` unless a config is passed in.
            This is only available in PyTorch 1.12 and later.
        activation_checkpointing: A single layer or a list of layer classes for which you want to enable activation
            checkpointing. This is typically your transformer block (including attention + feed-forward).
            Enabling this can free up a significant amount of memory at the cost of speed since activations in
            these layers need to be recomputed during backpropagation.
        \**kwargs: Passed to the FSDP context manager which will configure the FSDP class when wrapping modules.

    """

    strategy_name = "fsdp_native"
    _registered_strategies: List[str] = []

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = None,
        cpu_offload: Union[bool, "CPUOffload", None] = None,
        backward_prefetch: Optional[BackwardPrefetch] = None,
        mixed_precision: Optional[MixedPrecision] = None,
        activation_checkpointing: Optional[Union[Type[Module], List[Type[Module]]]] = None,
        **kwargs: Any,
    ) -> None:
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
        self.cpu_offload = _init_cpu_offload(cpu_offload)
        self.backward_prefetch = backward_prefetch
        self.mixed_precision = mixed_precision
        self._rank_0_will_call_children_scripts: bool = False
        if activation_checkpointing and not _TORCH_GREATER_EQUAL_1_13:
            raise ValueError("Activation checkpointing requires torch >= 1.13.0. HINT: `pip install -U torch`")
        activation_checkpointing = activation_checkpointing or []
        self._activation_checkpointing = (
            [activation_checkpointing] if not isinstance(activation_checkpointing, list) else activation_checkpointing
        )
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
        _init_dist_connection(self.cluster_environment, self._process_group_backend)
        super().setup_environment()

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

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

    def _setup_model(self, model: torch.nn.Module) -> FullyShardedDataParallel:
        """Wraps the model into a
        :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel` module."""
        # If model is already wrapped, we need to avoid sending the `auto_wrap_policy`
        assert self.lightning_module is not None
        if "auto_wrap_policy" in self.kwargs and any(
            isinstance(mod, FullyShardedDataParallel) for mod in self.lightning_module.modules()
        ):
            del self.kwargs["auto_wrap_policy"]

        log.detail(f"setting up FSDP model with device id: {self.root_device.index}, kwargs: {self.kwargs}")

        wrapped_module = FullyShardedDataParallel(
            module=model,
            process_group=self.process_group,
            cpu_offload=self.cpu_offload,
            backward_prefetch=self.backward_prefetch,
            mixed_precision=self.mixed_precision_config,
            device_id=self.root_device.index,
            **self.kwargs,
        )

        # activation checkpointing needs to be set up after wrapping the model
        if _TORCH_GREATER_EQUAL_1_13 and self._activation_checkpointing:
            _setup_activation_checkpointing(module=wrapped_module, layers=self._activation_checkpointing)

        return wrapped_module

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

        assert isinstance(self.model, pl.LightningModule)
        self.model = _LightningModuleWrapperBase(self.model)
        if is_overridden("configure_sharded_model", self.lightning_module):
            rank_zero_info(
                "You have overridden `LightningModule.configure_sharded_model` hook. It will assume that all the layers"
                " are already wrapped for sharding and won't wrap the entire model using `FullyShardedDataParallel`."
            )
        else:
            self.model = self._setup_model(self.model)
        self.barrier()

        self.setup_optimizers(trainer)
        _optimizers_to_device(self.optimizers, self.root_device)

        self.setup_precision_plugin()

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        invalid_params_error = False
        try:
            super().setup_optimizers(trainer)
        except ValueError as e:
            if "optimizer got an empty parameter list" not in str(e):
                raise
            invalid_params_error = True

        if invalid_params_error or any(not _optimizer_has_flat_params(optimizer) for optimizer in self.optimizers):
            raise ValueError(
                "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the"
                " optimizer after setting up the model by referencing `self.trainer.model.parameters()` in the"
                " `configure_optimizers()` hook."
            )

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
            tensor = _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # we don't need precision context since casting is done by FSDP
        # read `mixed_precision` docstring here: https://pytorch.org/docs/stable/fsdp.html
        assert self.model is not None
        return self.model(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        return self.model(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        return self.model(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.model is not None
        return self.model(*args, **kwargs)

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
                cpu_offload=True,
            )
            cls._registered_strategies.append("fsdp_native_full_shard_offload")
