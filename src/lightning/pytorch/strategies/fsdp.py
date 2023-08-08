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
import logging
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Any, Callable, Dict, Generator, List, Literal, Mapping, Optional, Set, Type, TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.strategies.fsdp import (
    _activation_checkpointing_kwargs,
    _auto_wrap_policy_kwargs,
    _get_full_state_dict_context,
    _init_cpu_offload,
    _init_sharding_strategy,
    _optimizer_has_flat_params,
    _setup_activation_checkpointing,
)
from lightning.fabric.utilities.distributed import (
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_1_12,
    _TORCH_GREATER_EQUAL_1_13,
    _TORCH_GREATER_EQUAL_2_0,
)
from lightning.fabric.utilities.init import _EmptyInit
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import ProcessGroup, ReduceOp
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.pytorch.plugins.precision.fsdp import FSDPMixedPrecisionPlugin
from lightning.pytorch.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy

    if _TORCH_GREATER_EQUAL_2_0:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        _POLICY = Union[Set, Callable[[Module, bool, int], bool], ModuleWrapPolicy]
    else:
        _POLICY = Union[Set, Callable[[Module, bool, int], bool]]  # type: ignore[misc]

    _SHARDING_STRATEGY = Union[ShardingStrategy, Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]]


log = logging.getLogger(__name__)


class FSDPStrategy(ParallelStrategy):
    r"""Strategy for Fully Sharded Data Parallel provided by torch.distributed.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Fully Sharded Training shards the entire model across all available GPUs, allowing you to scale model
    size, whilst using efficient communication to reduce overhead. In practice, this means we can remain
    at parity with PyTorch DDP, whilst scaling our model sizes dramatically. The technique is similar
    to ZeRO-Stage 3.

    For more information check out
    `this blogpost <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api>`__.

    Defaults have been set and options have been exposed, but may require configuration
    based on your level of memory/speed efficiency. We suggest having a look at
    `this tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__ for more information.

    Arguments:
        cpu_offload: See ``cpu_offload`` parameter in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.
        mixed_precision: See ``mixed_precision`` parameter in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.
        auto_wrap_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch.distributed.fsdp.FullyShardedDataParallel`. For convenience, this also accepts a set of the
            layer classes to wrap.
        activation_checkpointing: Deprecated. Use ``activation_checkpointing_policy``.
        activation_checkpointing_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch.distributed.fsdp.FullyShardedDataParallel` but used when selecting the modules for which you
            want to enable activation checkpointing. Enabling this can free up a significant amount of memory at the
            cost of speed since activations in these layers need to be recomputed during backpropagation. For
            convenience, this also accepts a set of the layer classes to wrap.
        sharding_strategy: Select whether to shard model parameters, gradients, optimizer states, or a combination of
            them. Available values are:

            - ``"FULL_SHARD"``: Shards model parameters, gradients, and optimizer states (default).
            - ``"SHARD_GRAD_OP"``: Shards gradients and optimizer states only. Model parameters get replicated.
            - ``"NO_SHARD"``: No sharding (identical to regular DDP).
            - ``"HYBRID_SHARD"``: Shards model parameters, gradients, and optimizer states within a single machine, but
              replicates across machines.

            Also accepts a :class:`torch.distributed.fsdp.ShardingStrategy` enum value.

        \**kwargs: See available parameters in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    """

    strategy_name = "fsdp"
    _registered_strategies: List[str] = []

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        cpu_offload: Union[bool, "CPUOffload", None] = None,
        mixed_precision: Optional["MixedPrecision"] = None,
        auto_wrap_policy: Optional["_POLICY"] = None,
        activation_checkpointing: Optional[Union[Type[Module], List[Type[Module]]]] = None,
        activation_checkpointing_policy: Optional["_POLICY"] = None,
        sharding_strategy: "_SHARDING_STRATEGY" = "FULL_SHARD",
        **kwargs: Any,
    ) -> None:
        if not _TORCH_GREATER_EQUAL_1_12:
            raise MisconfigurationException("`FSDPStrategy` is supported from PyTorch v1.12.0 onwards.")

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
        self._timeout: Optional[timedelta] = timeout
        self.cpu_offload = _init_cpu_offload(cpu_offload)
        self.sharding_strategy = _init_sharding_strategy(sharding_strategy)
        self.mixed_precision = mixed_precision
        self.kwargs = _auto_wrap_policy_kwargs(auto_wrap_policy, kwargs)

        if _TORCH_GREATER_EQUAL_2_0:
            # Avoids the need for user to reference params in `configure_optimizers` via
            # `self.trainer.model.parameters()` and enables support for multiple parameter groups.
            self.kwargs.setdefault("use_orig_params", True)

        self._activation_checkpointing_kwargs = _activation_checkpointing_kwargs(
            activation_checkpointing, activation_checkpointing_policy
        )

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """Gathers the full state dict by unsharding all the parameters.

        To avoid OOM, the returned parameters will only be returned on rank 0 and on CPU. All other ranks get an empty
        dict.

        """
        from torch.distributed.fsdp import FullyShardedDataParallel
        from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

        assert self.model is not None

        with FullyShardedDataParallel.state_dict_type(
            module=self.model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=(self.world_size > 1), rank0_only=True),
        ):
            return self.model.state_dict()

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
            from torch.distributed.distributed_c10d import _get_default_group

            # The strategy should have already initilized process group in setup_environment()
            self._process_group = _get_default_group()
        return self._process_group

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    def mixed_precision_config(self) -> Optional["MixedPrecision"]:
        if self.mixed_precision:
            return self.mixed_precision
        plugin = self.precision_plugin
        if isinstance(plugin, FSDPMixedPrecisionPlugin):
            return plugin.mixed_precision_config
        return None

    @property
    def distributed_sampler_kwargs(self) -> Dict:
        return {"num_replicas": (self.num_nodes * self.num_processes), "rank": self.global_rank}

    def setup_environment(self) -> None:
        log.debug(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)
        super().setup_environment()

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = self.global_rank

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    def _setup_model(self, model: Module) -> Module:
        """Wraps the model into a
        :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel` module."""
        from torch.distributed.fsdp import FullyShardedDataParallel

        if any(isinstance(mod, FullyShardedDataParallel) for mod in model.modules()):
            # the user wrapped at least one layer in `configure_model` already
            if "auto_wrap_policy" in self.kwargs:
                rank_zero_warn(
                    "A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored."
                )
                del self.kwargs["auto_wrap_policy"]
        else:
            log.debug(f"setting up FSDP model with device id: {self.root_device.index}, kwargs: {self.kwargs}")
            model = FullyShardedDataParallel(
                module=model,
                process_group=self.process_group,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision_config,
                sharding_strategy=self.sharding_strategy,
                device_id=self.root_device.index,
                **self.kwargs,
            )

        # activation checkpointing needs to be set up after wrapping the model
        if _TORCH_GREATER_EQUAL_1_13:
            _setup_activation_checkpointing(model, self._activation_checkpointing_kwargs)

        return model

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        assert self.model is not None
        self.accelerator.setup(trainer)

        if trainer.state.fn == TrainerFn.FITTING and self._layer_sync:
            self.model = self._layer_sync.apply(self.model)

        # we set the device so that optimizers can be created with distributed comms.
        assert self.lightning_module is not None
        self.lightning_module._device = self.root_device

        if is_overridden("configure_sharded_model", self.lightning_module):
            # legacy: we don't skip setup with the `configure_model` alternative
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
        if self.kwargs.get("use_orig_params"):
            return super().setup_optimizers(trainer)

        invalid_params_error = False
        try:
            super().setup_optimizers(trainer)
        except ValueError as ex:
            if "optimizer got an empty parameter list" not in str(ex):
                raise
            invalid_params_error = True

        if invalid_params_error or any(not _optimizer_has_flat_params(optimizer) for optimizer in self.optimizers):
            # We avoid this limitation in PyTorch >= 2.0 by setting `use_orig_params=True`
            raise ValueError(
                "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the"
                " optimizer after setting up the model by referencing `self.trainer.model.parameters()` in the"
                " `configure_optimizers()` hook."
            )
        return None

    def model_to_device(self) -> None:
        pass

    @contextmanager
    def tensor_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        # TODO: Use the meta device and reset parameters after https://github.com/pytorch/pytorch/issues/90465
        # is resolved. For now, the module will get moved to the device in `setup_module`.
        empty_init_context = _EmptyInit(enabled=bool(empty_init)) if _TORCH_GREATER_EQUAL_1_13 else nullcontext()
        with empty_init_context, self.precision_plugin.init_context():
            yield

    @contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        log.debug(f"{self.__class__.__name__}: entered model_sharded_context.")
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import enable_wrap

        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            process_group=self.process_group,
            cpu_offload=self.cpu_offload,
            mixed_precision=self.mixed_precision_config,
            sharding_strategy=self.sharding_strategy,
            device_id=self.root_device.index,
            **self.kwargs,
        ):
            yield

    def barrier(self, name: Optional[str] = None) -> None:
        if not torch.distributed.is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not torch.distributed.is_initialized():
            return obj

        obj = [obj]
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
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
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
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if not _TORCH_GREATER_EQUAL_1_12 or not torch.distributed.is_available():
            return
        strategy_registry.register(
            "fsdp",
            cls,
            description="Fully Sharded Data Parallel (FSDP) training",
        )
        cls._registered_strategies.append("fsdp")

        strategy_registry.register(
            "fsdp_cpu_offload",
            cls,
            description="Fully Sharded Data Parallel (FSDP) training with Full Sharding and CPU Offloading",
            cpu_offload=True,
        )
        cls._registered_strategies.append("fsdp_cpu_offload")

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        from torch.distributed.fsdp import FullyShardedDataParallel, OptimStateKeyType

        optimizer_states = checkpoint.get("optimizer_states")

        # If the optimizer states are not present, we don't need to do anything (backward compatibility)
        if optimizer_states is None:
            return

        if len(self.optimizers) != len(optimizer_states):
            raise RuntimeError(
                f"You have configured {len(self.optimizers)} optimizers but the checkpoint contains"
                f" {len(optimizer_states)} optimizers to load. Please resume training with the same number"
                " of optimizers or edit the checkpoint manually to remove states."
            )

        assert self.model is not None

        # rank0_only should be false because we need to load the optimizer state on all ranks
        with _get_full_state_dict_context(self.model, rank0_only=False):
            for optimizer, opt_state in zip(self.optimizers, optimizer_states):
                # convert the optimizer state to the format expected by FSDP
                opt_state = FullyShardedDataParallel.rekey_optim_state_dict(
                    opt_state, OptimStateKeyType.PARAM_NAME, self.model
                )

                opt_state = FullyShardedDataParallel.optim_state_dict_to_load(
                    optim_state_dict=opt_state,
                    model=self.model,
                    optim=optimizer,
                )

                optimizer.load_state_dict(opt_state)

    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        from torch.distributed.fsdp import FullyShardedDataParallel, OptimStateKeyType

        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer

        assert self.model is not None

        with _get_full_state_dict_context(self.model, rank0_only=True):
            state_dict = FullyShardedDataParallel.optim_state_dict(self.model, optimizer)

        # Store the optimizer state dict in standard format
        if self.global_rank == 0:
            state_dict = FullyShardedDataParallel.rekey_optim_state_dict(
                state_dict, OptimStateKeyType.PARAM_ID, self.model
            )

        return state_dict
