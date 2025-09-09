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
import shutil
from collections.abc import Generator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
)

import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.strategies.fsdp import (
    _METADATA_FILENAME,
    _distributed_checkpoint_load,
    _distributed_checkpoint_save,
    _get_full_state_dict_context,
    _get_sharded_state_dict_context,
    _is_full_checkpoint,
    _is_sharded_checkpoint,
    _move_torchmetrics_to_device,
    _optimizer_has_flat_params,
    _setup_activation_checkpointing,
)
from lightning.fabric.strategies.model_parallel import _load_raw_module_state
from lightning.fabric.utilities.distributed import (
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.fabric.utilities.init import _has_meta_device_parameters_or_buffers
from lightning.fabric.utilities.load import _lazy_load, _materialize_tensors
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, ReduceOp
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.plugins.precision.fsdp2 import FSDP2Precision
from lightning.pytorch.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    _POLICY = Union[set[type[Module]], Callable[[Module, bool, int], bool], ModuleWrapPolicy]
    _SHARDING_STRATEGY = Union[ShardingStrategy, Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]]


log = logging.getLogger(__name__)


class FSDP2Strategy(ParallelStrategy):
    r"""Strategy for Fully Sharded Data Parallel v2 (FSDP2) provided by ``torch.distributed``.

    FSDP2 is the next-generation implementation of Fully Sharded Data Parallel, built on top of the
    ``DeviceMesh`` and ``DTensor`` abstractions. It provides a more robust and extensible way of
    scaling models across devices, addressing many of the limitations and inconsistencies of the
    original FSDP (referred to here as FSDP1).

    Compared to FSDP1, FSDP2 offers:
      - Deterministic and composable sharding plans via ``DeviceMesh``
      - A unified tensor abstraction (``DTensor``) that enables interoperability between FSDP,
        tensor parallelism, and pipeline parallelism
      - Cleaner checkpointing semantics, reducing many of the loading/saving issues seen in FSDP1
      - Forward compatibility, as PyTorch maintainers are actively deprecating FSDP1 in favor of FSDP2

    For background, see the RFC:
    https://github.com/pytorch/pytorch/issues/114299

    Arguments:
        device_mesh: A :class:`torch.distributed.device_mesh.DeviceMesh` object that specifies
            how devices are arranged and how tensors should be sharded/replicated.
        parallelize_module: Optional policy function or mapping that specifies how to wrap or
            distribute submodules of the model using ``DTensor``.
        checkpoint_policy: Defines how checkpoint saving/loading is performed with DTensor-based
            modules. See ``torch.distributed.checkpoint`` for available options.
        mixed_precision: Optional policy for mixed precision training. Can be used to specify
            precision for parameters, gradients, and buffers.
        \**kwargs: Additional keyword arguments passed to the underlying FSDP2 APIs.

    .. note::
        FSDP2 is still marked as "not fully stable" in PyTorch, but it is the recommended path
        forward. FSDP1 will eventually be deprecated. Users are encouraged to migrate to FSDP2
        for new projects, but should test thoroughly before deploying in production-critical
        environments.

    """

    strategy_name = "fsdp2"
    _registered_strategies: list[str] = []

    def __init__(
        self,
        device_mesh: Union[tuple[int], "DeviceMesh"] = None,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        cpu_offload: Union[bool, "CPUOffloadPolicy", None] = None,
        mp_policy: Optional["MixedPrecisionPolicy"] = None,
        state_dict_type: Literal["full", "sharded"] = "full",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self.num_nodes = 1
        self._process_group_backend = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self.cpu_offload = _init_fsdp2_cpu_offload(cpu_offload)
        self.mp_policy = _init_fsdp2_mp_policy(mp_policy)

        self.device_mesh = device_mesh
        self._state_dict_type = state_dict_type

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    @override
    def precision_plugin(self) -> FSDP2Precision:
        plugin = self._precision_plugin
        if plugin is not None:
            assert isinstance(plugin, FSDP2Precision)
            return plugin
        return FSDP2Precision("32-true")

    @precision_plugin.setter
    @override
    def precision_plugin(self, precision_plugin: Optional[Precision]) -> None:
        if precision_plugin is not None and not isinstance(precision_plugin, FSDP2Precision):
            raise TypeError(
                f"The FSDP2 strategy can only work with the `FSDP2Precision` plugin, found {precision_plugin}"
            )
        self._precision_plugin = precision_plugin

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict:
        return {"num_replicas": (self.num_nodes * self.num_processes), "rank": self.global_rank}

    @property
    @override
    def restore_checkpoint_after_setup(self) -> bool:
        return True

    @property
    @override
    def lightning_restore_optimizer(self) -> bool:
        return False

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        log.debug(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if _TORCH_GREATER_EQUAL_2_3:
            kwargs["device_id"] = self.root_device if self.root_device.type != "cpu" else None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, **kwargs)

        # if device_mesh is None, get the world_size and create a 1D device mesh
        if self.device_mesh is None:
            world_size = self.cluster_environment.world_size()
            self.device_mesh = (world_size,)  # a 1-D tuple
        # if 'device_mesh' is provided as a tuple, update it into the `DeviceMesh` object here
        if isinstance(self.device_mesh, tuple):
            from torch.distributed.device_mesh import init_device_mesh

            self.device_mesh = init_device_mesh("cuda", self.device_mesh)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def _setup_model(self, model: Module) -> Module:
        """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module."""
        from torch.distributed.fsdp import FullyShardedDataParallel

        if any(isinstance(mod, FullyShardedDataParallel) for mod in model.modules()):
            if _has_meta_device_parameters_or_buffers(model):
                rank_zero_warn(
                    "The model is already wrapped in `FSDP` but there are still parameters on the meta device."
                )
            if "auto_wrap_policy" in self.kwargs:
                # The user has wrapped their submodules manually, don't apply the auto wrap policy.
                rank_zero_warn(
                    "A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored."
                )
                del self.kwargs["auto_wrap_policy"]
        else:
            log.debug(f"setting up FSDP model with device id: {self.root_device.index}, kwargs: {self.kwargs}")
            model = FullyShardedDataParallel(
                module=model,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision_config,
                sharding_strategy=self.sharding_strategy,
                device_id=self.root_device.index,
                **self.kwargs,
            )

        _move_torchmetrics_to_device(model, self.root_device)

        # activation checkpointing needs to be set up after wrapping the model
        _setup_activation_checkpointing(model, self._activation_checkpointing_kwargs)

        return model

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        assert self.model is not None
        if trainer.state.fn == TrainerFn.FITTING and self._layer_sync:
            self.model = self._layer_sync.apply(self.model)

        self.model = self.precision_plugin.convert_module(self.model)

        if is_overridden("configure_sharded_model", self.lightning_module):
            # legacy: we don't skip setup with the `configure_model` alternative
            rank_zero_info(
                "You have overridden `LightningModule.configure_sharded_model` hook. It will assume that all the layers"
                " are already wrapped for sharding and won't wrap the entire model using `FullyShardedDataParallel`."
            )
        else:
            self.model = self._setup_model(self.model)
        self.barrier()

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        if trainer.state.fn == TrainerFn.FITTING:
            _optimizers_to_device(self.optimizers, self.root_device)

    @override
    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        # If we're setting up for evaluation after fitting, we need to discard the optimizers
        # since we're rewrapping the model, otherwise optimizer param references are no longer valid
        # and subsequent checkpoint saving can fail
        self._reset_optimizers_and_schedulers()

        if self.kwargs.get("use_orig_params"):
            return super().setup_optimizers(trainer)

        invalid_params_error = False
        try:
            # If `use_orig_params=False` the user needs to do access `self.trainer.model.parameters()` in
            # `configure_optimizers()`
            super().setup_optimizers(trainer)
        except ValueError as ex:
            if "optimizer got an empty parameter list" not in str(ex):
                raise
            invalid_params_error = True

        if invalid_params_error or any(not _optimizer_has_flat_params(optimizer) for optimizer in self.optimizers):
            # We avoid this limitation by setting `use_orig_params=True`
            raise ValueError(
                "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the"
                " optimizer after setting up the model by referencing `self.trainer.model.parameters()` in the"
                " `configure_optimizers()` hook."
            )
        return None

    @override
    def model_to_device(self) -> None:
        # FSDP takes care of moving the model to device
        pass

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        # Materialization happens in `setup`. When modules get wrapped by FSDP, the sequence of operations is:
        # 1) materialize module 2) call `reset_parameters()` 3) shard the module.
        # These operations are applied to each submodule 'bottom up' in the module hierarchy.
        empty_init_context = torch.device("meta") if empty_init else nullcontext()
        with empty_init_context, self.precision_plugin.tensor_init_context():
            yield

    @contextmanager
    @override
    def model_sharded_context(self) -> Generator[None, None, None]:
        log.debug(f"{self.__class__.__name__}: entered model_sharded_context.")
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import enable_wrap

        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            cpu_offload=self.cpu_offload,
            mixed_precision=self.mixed_precision_config,
            sharding_strategy=self.sharding_strategy,
            device_id=self.root_device.index,
            **self.kwargs,
        ):
            yield

    @override
    def barrier(self, name: Optional[str] = None) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self._determine_device_ids())
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @override
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

    def _determine_device_ids(self) -> list[int]:
        return [self.root_device.index]

    @override
    def teardown(self) -> None:
        log.debug(f"{self.__class__.__name__}: tearing down strategy...")

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
    def get_registered_strategies(cls) -> list[str]:
        return cls._registered_strategies

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if not torch.distributed.is_available():
            return
        strategy_registry.register(
            "fsdp2",
            cls,
            description="FSDP2 training",
        )
        cls._registered_strategies.append("fsdp2")

        strategy_registry.register(
            "fsdp_cpu_offload",
            cls,
            description="FSDP2 training with Full Sharding and CPU Offloading",
            cpu_offload=True,
        )
        cls._registered_strategies.append("fsdp2_cpu_offload")

    @override
    def lightning_module_state_dict(self) -> dict[str, Any]:
        assert self.model is not None
        if self._state_dict_type == "sharded":
            state_dict_ctx = _get_sharded_state_dict_context(self.model)
        elif self._state_dict_type == "full":
            state_dict_ctx = _get_full_state_dict_context(self.model, world_size=self.world_size)
        else:
            raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")
        with state_dict_ctx:
            return self.model.state_dict()

    @override
    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        # Override to do nothing, FSDP already loaded the states in `load_checkpoint()`
        pass

    @override
    def optimizer_state(self, optimizer: Optimizer) -> dict[str, Tensor]:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import OptimStateKeyType

        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer

        assert self.model is not None
        if self._state_dict_type == "sharded":
            with _get_sharded_state_dict_context(self.model):
                return FSDP.optim_state_dict(self.model, optimizer)

        elif self._state_dict_type == "full":
            with _get_full_state_dict_context(self.model, world_size=self.world_size):
                state_dict = FSDP.optim_state_dict(self.model, optimizer)
                if self.global_rank == 0:
                    # Store the optimizer state dict in standard format
                    state_dict = FSDP.rekey_optim_state_dict(state_dict, OptimStateKeyType.PARAM_ID, self.model)
                return state_dict

        raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    @override
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Override to do nothing, the FSDP already loaded the states in `load_checkpoint()`
        pass

    @override
    def save_checkpoint(
        self, checkpoint: dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `FSDPStrategy` does not use the `CheckpointIO`."
            )

        path = Path(self.broadcast(filepath))
        if path.is_dir() and self._state_dict_type == "full" and not _is_sharded_checkpoint(path):
            raise IsADirectoryError(f"The checkpoint path exists and is a directory: {path}")

        if self._state_dict_type == "sharded":
            if path.is_file():
                path.unlink()
            path.mkdir(parents=True, exist_ok=True)

            converted_state = {"model": checkpoint.pop("state_dict")}
            converted_state.update({
                f"optimizer_{idx}": optim_state
                for idx, optim_state in enumerate(checkpoint.pop("optimizer_states", []))
            })

            _distributed_checkpoint_save(converted_state, path)

            if self.global_rank == 0:
                torch.save(checkpoint, path / _METADATA_FILENAME)
        elif self._state_dict_type == "full":
            if _is_sharded_checkpoint(path):
                shutil.rmtree(path)
            return super().save_checkpoint(checkpoint=checkpoint, filepath=path)
        else:
            raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    @override
    def load_checkpoint(self, checkpoint_path: _PATH) -> dict[str, Any]:
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(checkpoint_path))

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        assert self.model is not None
        assert self.lightning_module is not None

        if _is_sharded_checkpoint(path):
            from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict

            state_dict_ctx = _get_sharded_state_dict_context(self.model)

            with state_dict_ctx:
                module_state = {"model": self.model.state_dict()}
                _distributed_checkpoint_load(module_state, path)
                self.model.load_state_dict(module_state["model"], strict=self.lightning_module.strict_loading)

                if self.lightning_module.trainer.state.fn == TrainerFn.FITTING and self.optimizers:
                    from torch.distributed.checkpoint import FileSystemReader

                    # TODO: replace with newer APIs
                    # https://github.com/pytorch/pytorch/issues/119800#issuecomment-1942156271
                    reader = FileSystemReader(path=path)
                    # the optimizer states must be loaded separately
                    for idx, optim in enumerate(self.optimizers):
                        optim_key = f"optimizer_{idx}"
                        optim_state = load_sharded_optimizer_state_dict(
                            model_state_dict=module_state["model"],
                            optimizer_key=optim_key,
                            storage_reader=reader,
                        )
                        flattened_osd = FSDP.optim_state_dict_to_load(
                            optim_state_dict=optim_state[optim_key],
                            model=self.model,
                            optim=optim,
                        )
                        optim.load_state_dict(flattened_osd)

            # Load metadata (anything not a module or optimizer)
            metadata = torch.load(path / _METADATA_FILENAME)
            return metadata

        if _is_full_checkpoint(path):
            checkpoint = _lazy_load(path)
            _load_raw_module_state(
                checkpoint.pop("state_dict"),
                module=self.model,
                world_size=self.world_size,
                strict=self.lightning_module.strict_loading,
            )

            # Materialize lazy tensors if there are any left in the checkpoint
            # The `torch.Optimizer.load_state_dict` method can't load lazy tensors because of deepcopy pickle issues
            checkpoint = _materialize_tensors(checkpoint)

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import OptimStateKeyType

            optimizer_states = checkpoint.get("optimizer_states")
            if optimizer_states is None or self.lightning_module.trainer.state.fn != TrainerFn.FITTING:
                # If the optimizer states are not present, we don't need to do anything (backward compatibility)
                return checkpoint
            if len(self.optimizers) != len(optimizer_states):
                raise RuntimeError(
                    f"You have configured {len(self.optimizers)} optimizers but the checkpoint contains"
                    f" {len(optimizer_states)} optimizers to load. Please resume training with the same number"
                    " of optimizers or edit the checkpoint manually to remove states."
                )

            # rank0_only should be false because we need to load the optimizer state on all ranks
            with _get_full_state_dict_context(self.model, world_size=self.world_size, rank0_only=False):
                for optimizer, opt_state in zip(self.optimizers, optimizer_states):
                    if isinstance(list(opt_state["state"].keys())[0], int):
                        # Handling the case where the optimizer state is saved from a normal optimizer
                        opt_state = FSDP.rekey_optim_state_dict(opt_state, OptimStateKeyType.PARAM_NAME, self.model)

                    opt_state = FSDP.optim_state_dict_to_load(
                        optim_state_dict=opt_state,
                        model=self.model,
                        optim=optimizer,
                    )
                    optimizer.load_state_dict(opt_state)

            return checkpoint

        raise ValueError(
            f"The path {str(path)!r} does not point to a valid checkpoint. Make sure the path points to either a"
            " directory with FSDP checkpoint shards, or a single file with a full checkpoint."
        )


def _init_fsdp2_cpu_offload(cpu_offload: Optional[Union[bool, "CPUOffloadPolicy"]]) -> "CPUOffloadPolicy":
    from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy

    if cpu_offload is None or cpu_offload is False:
        return OffloadPolicy()

    if cpu_offload is True:
        return CPUOffloadPolicy(pin_memory=True)

    if isinstance(cpu_offload, CPUOffloadPolicy):
        return cpu_offload

    raise TypeError(f"`cpu_offload` should be of type `bool` or `CPUOffloadPolicy`, got {type(cpu_offload)}")


def _init_fsdp2_mp_policy(mp_policy: Optional["MixedPrecisionPolicy"]) -> Optional["MixedPrecisionPolicy"]:
    from torch.distributed.fsdp import MixedPrecisionPolicy

    if mp_policy is None:
        return MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=True)

    if isinstance(mp_policy, MixedPrecisionPolicy):
        return mp_policy

    raise TypeError(f"`mp_policy` should be of type `MixedPrecisionPolicy`, got {type(mp_policy)}")
