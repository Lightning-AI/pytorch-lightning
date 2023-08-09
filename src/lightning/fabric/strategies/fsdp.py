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
import threading
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from typing_extensions import TypeGuard

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.precision.fsdp import FSDPPrecision
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import (
    _apply_filter,
    _BackwardSyncControl,
    _Sharded,
    _validate_keys_for_strict_loading,
    TBroadcast,
)
from lightning.fabric.utilities.distributed import (
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.distributed import ReduceOp
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_1_12,
    _TORCH_GREATER_EQUAL_1_13,
    _TORCH_GREATER_EQUAL_2_0,
    _TORCH_GREATER_EQUAL_2_1,
)
from lightning.fabric.utilities.init import _EmptyInit
from lightning.fabric.utilities.load import _lazy_load, _materialize_tensors
from lightning.fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy

    from lightning.fabric.wrappers import _FabricModule

    if _TORCH_GREATER_EQUAL_2_0:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        _POLICY = Union[Set, Callable[[Module, bool, int], bool], ModuleWrapPolicy]
    else:
        _POLICY = Union[Set, Callable[[Module, bool, int], bool]]  # type: ignore[misc]

    _SHARDING_STRATEGY = Union[ShardingStrategy, Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]]

_FSDP_ALIASES = ("fsdp", "fsdp_cpu_offload")
_METADATA_FILENAME = "meta.pt"


class FSDPStrategy(ParallelStrategy, _Sharded):
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

        state_dict_type: The format in which the state of the model and optimizers gets saved into the checkpoint.

            - ``"full"``: The full weights and optimizer states get assembled on rank 0 and saved to a single file.
            - ``"sharded"``: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
              a folder with as many files as the world size.

        \**kwargs: See available parameters in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    """

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        cpu_offload: Union[bool, "CPUOffload", None] = None,
        mixed_precision: Optional["MixedPrecision"] = None,
        auto_wrap_policy: Optional["_POLICY"] = None,
        activation_checkpointing: Optional[Union[Type[Module], List[Type[Module]]]] = None,
        activation_checkpointing_policy: Optional["_POLICY"] = None,
        sharding_strategy: "_SHARDING_STRATEGY" = "FULL_SHARD",
        state_dict_type: Literal["full", "sharded"] = "sharded",
        **kwargs: Any,
    ) -> None:
        if not _TORCH_GREATER_EQUAL_1_12:
            raise NotImplementedError("`FSDPStrategy` is supported from PyTorch v1.12.0 onwards.")

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            precision=precision,
        )
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._backward_sync_control = _FSDPBackwardSyncControl()
        self._fsdp_kwargs = _auto_wrap_policy_kwargs(auto_wrap_policy, kwargs)

        if _TORCH_GREATER_EQUAL_2_0:
            # Enables joint setup of model and optimizer, multiple optimizer param groups, and `torch.compile()`
            self._fsdp_kwargs.setdefault("use_orig_params", True)

        self._activation_checkpointing_kwargs = _activation_checkpointing_kwargs(
            activation_checkpointing, activation_checkpointing_policy
        )
        self._state_dict_type = state_dict_type
        self.sharding_strategy = _init_sharding_strategy(sharding_strategy)
        self.cpu_offload = _init_cpu_offload(cpu_offload)
        self.mixed_precision = mixed_precision

    @property
    def checkpoint_io(self) -> CheckpointIO:
        raise NotImplementedError(f"The `{type(self).__name__}` does not use the `CheckpointIO` plugin interface.")

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        raise NotImplementedError(f"The `{type(self).__name__}` does not support setting a `CheckpointIO` plugin.")

    @property
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        return {"num_replicas": (self.num_nodes * self.num_processes), "rank": self.global_rank}

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    def mixed_precision_config(self) -> Optional["MixedPrecision"]:
        if self.mixed_precision:
            return self.mixed_precision
        if isinstance(self.precision, FSDPPrecision):
            return self.precision.mixed_precision_config
        return None

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    def setup_environment(self) -> None:
        self._setup_distributed()
        super().setup_environment()

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model into a
        :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel` module
        and sets `use_orig_params=True` to keep the reference to the original parameters in the
        optimizer.
        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                f"The `{type(self).__name__}` does not support the joint setup of module and optimizer(s)."
                " Please do it in this order: Create the model, call `setup_module`, create the optimizer,"
                " call `setup_optimizer`."
            )
        use_orig_params = self._fsdp_kwargs.get("use_orig_params")
        if use_orig_params is False:
            raise ValueError(
                f"You set `{type(self).__name__}(use_orig_params=False)` but this is not supported when"
                " setting the model and optimizer up jointly. Either set it to `True` or set the objects"
                " up in this order: Create the model, call `setup_module`, create the optimizer,"
                " call `setup_optimizer`."
            )
        module = self.setup_module(module)
        return module, optimizers

    def setup_module(self, module: Module) -> Module:
        """Wraps the model into a
        :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel` module."""
        from torch.distributed.fsdp import FullyShardedDataParallel

        if any(isinstance(mod, FullyShardedDataParallel) for mod in module.modules()):
            # The user has wrapped their submodules manually, don't apply the auto wrap policy.
            if _has_meta_device_parameters(module):
                rank_zero_warn(
                    "The model is already wrapped in `FSDP` but there are still parameters on the meta device."
                )
            if "auto_wrap_policy" in self._fsdp_kwargs:
                rank_zero_warn(
                    "A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored."
                )
                del self._fsdp_kwargs["auto_wrap_policy"]
        else:
            module = FullyShardedDataParallel(
                module=module,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision_config,
                sharding_strategy=self.sharding_strategy,
                device_id=self.root_device.index,
                **self._fsdp_kwargs,
            )

        # activation checkpointing needs to be set up after wrapping the model
        if _TORCH_GREATER_EQUAL_1_13:
            _setup_activation_checkpointing(module, self._activation_checkpointing_kwargs)

        return module

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Set up an optimizer for a model wrapped with FSDP.

        This setup method doesn't modify the optimizer or wrap the optimizer. The only thing it currently does is verify
        that the optimizer was created after the model was wrapped with :meth:`setup_module` with a reference to the
        flattened parameters.

        """
        if _TORCH_GREATER_EQUAL_2_0:
            return optimizer

        from torch.distributed.fsdp import FlatParameter

        num_groups = len(optimizer.param_groups)
        if num_groups > 1:
            raise ValueError(
                "An optimizer used with an FSDP model does not support multiple param groups."
                f" Found {num_groups} parameter groups."
            )

        if any(isinstance(param, FlatParameter) for param in optimizer.param_groups[0]["params"]):
            return optimizer

        raise ValueError(
            "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the optimizer"
            " after setting up the model."
        )

    def module_to_device(self, module: Module) -> None:
        pass

    @contextmanager
    def module_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        empty_init_context: Union[torch.device, _EmptyInit, nullcontext]
        if _TORCH_GREATER_EQUAL_2_1 and empty_init:
            # Materialization happens in `setup`. When modules get wrapped by FSDP, the sequence of operations is:
            # 1) materialize module 2) call `reset_parameters()` 3) shard the module.
            # These operations are applied to each submodule 'bottom up' in the module hierarchy.
            empty_init_context = torch.device("meta")
        elif _TORCH_GREATER_EQUAL_1_13:
            empty_init_context = _EmptyInit(enabled=bool(empty_init))
        else:
            empty_init_context = nullcontext()
        with empty_init_context, self.precision.init_context(), self.module_sharded_context():
            yield

    @contextmanager
    def module_sharded_context(self) -> Generator:
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import enable_wrap

        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            cpu_offload=self.cpu_offload,
            mixed_precision=self.mixed_precision_config,
            sharding_strategy=self.sharding_strategy,
            device_id=self.root_device.index,
            **self._fsdp_kwargs,
        ):
            yield

    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not torch.distributed.is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[self.root_device.index])
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not torch.distributed.is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        """Clip gradients by norm."""
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

        if not isinstance(module, FullyShardedDataParallel):
            # the root must be wrapped
            raise TypeError(
                "Gradient clipping with FSDP is only possible if the module passed to"
                f" `{self.__class__.__name__}.clip_gradients_norm` is wrapped in `FullyShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        self.precision.unscale_gradients(optimizer)
        return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)

    def clip_gradients_value(self, module: Module, optimizer: Optimizer, clip_val: Union[float, int]) -> None:
        """Clip gradients by value."""
        raise NotImplementedError(
            "FSDP currently does not support to clip gradients by value. "
            "Consider clipping by norm instead or choose another strategy!"
        )

    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state to a checkpoint on disk.

        If the state-dict-type is ``'full'``, the checkpoint will be written to a single file containing the weights,
        optimizer state and other metadata. If the state-dict-type is ``'sharded'``, the checkpoint gets saved as a
        directory containing one file per process, with model- and optimizer shards stored per file. Additionally, it
        creates a metadata file `meta.pt` with the rest of the user's state (only saved from rank 0).

        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `FSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`."
            )
        if storage_options is not None:
            raise TypeError(
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `FSDPStrategy` does not use the `CheckpointIO`."
            )
        if filter is not None and self._state_dict_type == "sharded":
            # https://github.com/pytorch/pytorch/issues/105379
            raise NotImplementedError(
                "FSDP doesn't support loading sharded filtered checkpoints, so saving them is disabled."
            )

        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(path))
        if path.is_dir() and os.listdir(path):
            raise FileExistsError(f"The checkpoint directory already exists and is not empty: {path}")

        from torch.distributed.checkpoint import FileSystemWriter, save_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        modules = [module for module in state.values() if _has_fsdp_modules(module)]
        if len(modules) == 0:
            raise ValueError(
                "Could not find a FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple FSDP models in the given state. Saving checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )
        module = modules[0]

        if self._state_dict_type == "sharded":
            path.mkdir(parents=True, exist_ok=True)
            state_dict_ctx = _get_sharded_state_dict_context(module)

            # replace the modules and optimizer objects in the state with their local state dict
            # and separate the user's metadata
            converted_state: Dict[str, Any] = {}
            metadata: Dict[str, Any] = {}
            with state_dict_ctx:
                for key, obj in state.items():
                    converted: Any
                    if isinstance(obj, Module):
                        converted = obj.state_dict()
                        target_dict = converted_state
                    elif isinstance(obj, Optimizer):
                        converted = FSDP.optim_state_dict(module, obj)
                        target_dict = converted_state
                    else:  # everything not a module or optimizer is considered metadata
                        converted = obj
                        target_dict = metadata
                    _apply_filter(key, filter or {}, converted, target_dict)

            # FSDP's FileSystemWriter streams the tensors to disk to minimize memory peaks
            writer = FileSystemWriter(path=path, single_file_per_rank=True)
            save_state_dict(converted_state, writer)

            if self.global_rank == 0:
                torch.save(metadata, path / _METADATA_FILENAME)

        elif self._state_dict_type == "full":
            state_dict_ctx = _get_full_state_dict_context(module)
            full_state: Dict[str, Any] = {}
            with state_dict_ctx:
                for key, obj in state.items():
                    if isinstance(obj, Module):
                        converted = obj.state_dict()
                    elif isinstance(obj, Optimizer):
                        converted = FSDP.optim_state_dict(module, obj)
                    else:  # everything not a module or optimizer is considered metadata
                        converted = obj
                    _apply_filter(key, filter or {}, converted, full_state)

            if self.global_rank == 0:
                torch.save(full_state, path)
        else:
            raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load the contents from a checkpoint and restore the state of the given objects.

        The strategy currently only supports saving and loading sharded checkpoints which are stored in form of a
        directory of multiple files rather than a single file.

        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `FSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`."
            )
        if not state:
            raise ValueError(
                f"Got FSDPStrategy.load_checkpoint(..., state={state!r}) but a state with at least "
                f" a model instance to reload is required. Pass it in like so:"
                " FSDPStrategy.load_checkpoint(..., state={'model': model, ...})"
            )
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(path))

        if isinstance(state, Module):
            _load_raw_module_state_from_path(path, module=state, strict=strict)
            return {}

        if isinstance(state, Optimizer):
            raise NotImplementedError(
                "Loading a single optimizer object from a checkpoint is not supported yet with the FSDP strategy."
            )

        from torch.distributed.checkpoint import FileSystemReader, load_state_dict
        from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import OptimStateKeyType

        modules = {key: module for key, module in state.items() if _has_fsdp_modules(module)}
        if len(modules) == 0:
            raise ValueError(
                "Could not find a FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before loading the checkpoint."
            )
        optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
        if len(modules) > 1:
            raise ValueError(
                "Found multiple FSDP models in the given state. Loading checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To load multiple models, call the"
                " load method for each model separately with a different path."
            )
        module_key, module = list(modules.items())[0]

        if _is_sharded_checkpoint(path):
            state_dict_ctx = _get_sharded_state_dict_context(module)
            reader = FileSystemReader(path=path)

            with state_dict_ctx:
                module_state = {module_key: module.state_dict()}
                load_state_dict(module_state, reader)
                module.load_state_dict(module_state[module_key], strict=strict)

                # the optimizer states must be loaded separately
                for optim_key, optim in optimizers.items():
                    optim_state = load_sharded_optimizer_state_dict(
                        model_state_dict=module_state[module_key],
                        optimizer_key=optim_key,
                        storage_reader=reader,
                    )
                    flattened_osd = FSDP.optim_state_dict_to_load(
                        optim_state_dict=optim_state[optim_key],
                        model=module,
                        optim=optim,
                    )
                    optim.load_state_dict(flattened_osd)

            # Load metadata (anything not a module or optimizer)
            metadata = torch.load(path / _METADATA_FILENAME)
            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, metadata.keys(), strict=strict)
            for key in requested_metadata_keys:
                if key not in metadata:
                    continue
                state[key] = metadata.pop(key)

            # return the remaining metadata that wasn't requested as part of `state`
            return metadata

        if _is_full_checkpoint(path):
            checkpoint = _lazy_load(path) if _TORCH_GREATER_EQUAL_2_0 else torch.load(path, map_location="cpu")
            _load_raw_module_state(checkpoint.pop(module_key), module=module, strict=strict)

            if isinstance(state, Module):
                return {}

            if _TORCH_GREATER_EQUAL_2_0:
                # Materialize lazy tensors if there are any left in the checkpoint
                # The `torch.Optimizer.load_state_dict` method can't load lazy tensors because of deepcopy pickle issues
                checkpoint = _materialize_tensors(checkpoint)

            # Load optimizer states
            for optim_key, optim in optimizers.items():
                # rank0_only should be false because we need to load the optimizer state on all ranks
                with _get_full_state_dict_context(module, rank0_only=False):
                    temp_state_dict = checkpoint.pop(optim_key)

                    # Handling the case where the optimizer state is saved from a normal optimizer
                    if isinstance(list(temp_state_dict["state"].keys())[0], int):
                        temp_state_dict = FSDP.rekey_optim_state_dict(
                            temp_state_dict, OptimStateKeyType.PARAM_NAME, module
                        )

                    optim_state_dict = FSDP.optim_state_dict_to_load(
                        optim_state_dict=temp_state_dict,
                        model=module,
                        optim=optim,
                    )
                    optim.load_state_dict(optim_state_dict)

            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, checkpoint.keys(), strict=strict)

            # Load metadata (anything not a module or optimizer)
            for key in requested_metadata_keys:
                if key not in checkpoint:
                    continue
                state[key] = checkpoint.pop(key)

            # return the remaining metadata that wasn't requested as part of `state`
            return checkpoint

        raise ValueError(
            f"The path {str(path)!r} does not point to a valid checkpoint. Make sure the path points to either a"
            " directory with FSDP checkpoint shards, or a single file with a full checkpoint."
        )

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if not _TORCH_GREATER_EQUAL_1_12 or not torch.distributed.is_available():
            return

        strategy_registry.register(
            "fsdp",
            cls,
            description="Fully Sharded Data Parallel (FSDP) training",
        )
        strategy_registry.register(
            "fsdp_cpu_offload",
            cls,
            description="Fully Sharded Data Parallel (FSDP) training with Full Sharding and CPU Offloading",
            cpu_offload=True,
        )

    def _setup_distributed(self) -> None:
        reset_seed()
        self._set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = self.global_rank


def _activation_checkpointing_kwargs(
    activation_checkpointing: Optional[Union[Type[Module], List[Type[Module]]]] = None,
    activation_checkpointing_policy: Optional["_POLICY"] = None,
) -> Dict:
    if activation_checkpointing is None and activation_checkpointing_policy is None:
        return {}
    if activation_checkpointing is not None and activation_checkpointing_policy is not None:
        raise ValueError(
            "You cannot set both `activation_checkpointing` and `activation_checkpointing_policy`. Use the latter."
        )
    if activation_checkpointing is not None:
        if not _TORCH_GREATER_EQUAL_1_13:
            raise ValueError("`activation_checkpointing` requires torch >= 1.13.0. HINT: `pip install -U torch`")
        if isinstance(activation_checkpointing, list):
            classes = tuple(activation_checkpointing)
        else:
            classes = (activation_checkpointing,)
        if _TORCH_GREATER_EQUAL_2_1:
            rank_zero_deprecation(
                f"`FSDPStrategy(activation_checkpointing={activation_checkpointing})` is deprecated, use "
                f"`FSDPStrategy(activation_checkpointing_policy={set(classes)})` instead."
            )
        return {"check_fn": lambda submodule: isinstance(submodule, classes)}
    if isinstance(activation_checkpointing_policy, set):
        if _TORCH_GREATER_EQUAL_2_1:
            return _auto_wrap_policy_kwargs(activation_checkpointing_policy, {})
        return {"check_fn": lambda submodule: isinstance(submodule, tuple(activation_checkpointing_policy))}
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ValueError("`activation_checkpointing_policy` requires torch >= 2.1.0. HINT: `pip install -U torch`")
    return {"auto_wrap_policy": activation_checkpointing_policy}


def _auto_wrap_policy_kwargs(policy: Optional["_POLICY"], kwargs: Dict) -> Dict:
    if policy is None:
        return kwargs
    if isinstance(policy, set):
        if _TORCH_GREATER_EQUAL_2_1:
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy

            policy = ModuleWrapPolicy(policy)
        else:
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            # this is not transformer specific despite the name
            policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=policy)
    kwargs["auto_wrap_policy"] = policy
    return kwargs


def _setup_activation_checkpointing(module: Module, activation_checkpointing_kwargs: Dict) -> None:
    if not activation_checkpointing_kwargs:
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    if any(isinstance(mod, CheckpointWrapper) for mod in module.modules()):
        rank_zero_warn(
            "FSDP checkpointing is configured, but the model already contains checkpointed layers."
            " Checkpointing will be ignored."
        )
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )

    wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(module, checkpoint_wrapper_fn=wrapper, **activation_checkpointing_kwargs)


class _FSDPBackwardSyncControl(_BackwardSyncControl):
    @contextmanager
    def no_backward_sync(self, module: Module) -> Generator:
        """Blocks gradient synchronization inside the
        :class:`~torch.distributed.fsdp.FullyShardedDataParallel` wrapper."""
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

        if not isinstance(module, FullyShardedDataParallel):
            # the root must be wrapped
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{self.__class__.__name__}.no_backward_sync` is wrapped in `FullyShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        with module.no_sync():
            yield


def _init_cpu_offload(cpu_offload: Optional[Union[bool, "CPUOffload"]]) -> "CPUOffload":
    from torch.distributed.fsdp import CPUOffload

    return cpu_offload if isinstance(cpu_offload, CPUOffload) else CPUOffload(offload_params=bool(cpu_offload))


def _init_sharding_strategy(sharding_strategy: "_SHARDING_STRATEGY") -> "ShardingStrategy":
    from torch.distributed.fsdp import ShardingStrategy

    return ShardingStrategy[sharding_strategy.upper()] if isinstance(sharding_strategy, str) else sharding_strategy


def _optimizer_has_flat_params(optimizer: Optimizer) -> bool:
    _FSDP_FLATTENED = "_fsdp_flattened"
    if _TORCH_GREATER_EQUAL_1_13:
        return any(
            getattr(param, _FSDP_FLATTENED, False) for group in optimizer.param_groups for param in group["params"]
        )

    from torch.distributed.fsdp import FlatParameter

    return any(isinstance(param, FlatParameter) for group in optimizer.param_groups for param in group["params"])


def _get_sharded_state_dict_context(module: Module) -> Generator[None, None, None]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType

    state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
    optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    state_dict_type_context = FSDP.state_dict_type(
        module=module,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    )
    return state_dict_type_context  # type: ignore[return-value]


def _get_full_state_dict_context(module: Module, rank0_only: bool = True) -> Generator[None, None, None]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType

    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
    optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
    state_dict_type_context = FSDP.state_dict_type(
        module=module,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    )
    return state_dict_type_context  # type: ignore[return-value]


def _is_sharded_checkpoint(path: Path) -> bool:
    """A heuristic check to determine whether the path points to a directory with checkpoint shards."""
    return path.is_dir() and (path / _METADATA_FILENAME).is_file()


def _is_full_checkpoint(path: Path) -> bool:
    return path.is_file()


def _has_fsdp_modules(module: object) -> TypeGuard[Module]:
    from torch.distributed.fsdp import FullyShardedDataParallel

    return isinstance(module, Module) and any(isinstance(m, FullyShardedDataParallel) for m in module.modules())


def _load_raw_module_state_from_path(path: Path, module: Module, strict: bool = True) -> None:
    """Loads the state dict from a file path into the FSDP module."""
    if not _is_full_checkpoint(path):
        raise ValueError(
            "Failed to load checkpoint directly into the model. The given path must be a single file containing the"
            f" full state dict: {path}"
        )
    # Use `lazy_load` instead of `torch.load` here to avoid storing a copy of the full checkpoint per rank
    _load_raw_module_state(state_dict=_lazy_load(path), module=module, strict=strict)


def _load_raw_module_state(state_dict: Dict[str, Any], module: Module, strict: bool = True) -> None:
    """Loads the state dict into the module by gathering all weights first and then and writing back to each shard."""

    with _get_full_state_dict_context(module, rank0_only=False):
        module.load_state_dict(state_dict, strict=strict)


def _has_meta_device_parameters(obj: Union[Module, Optimizer]) -> bool:
    if isinstance(obj, Optimizer):
        return any(
            t.is_meta for param_group in obj.param_groups for t in param_group["params"] if isinstance(t, Parameter)
        )
    if isinstance(obj, Module):
        return any(t.is_meta for t in obj.parameters())
    raise TypeError(f"Expected `torch.nn.Module` or `torch.optim.Optimizer`, got: {type(obj).__name__}")


def _no_op() -> None:
    pass


@contextmanager
def _apply_optimizers_during_fsdp_backward(
    optimizers: Union[Optimizer, Iterable[Optimizer]],
    module: Module,
) -> Generator[None, None, None]:
    """Call `Optimizer.step` as gradients become available.

    NOTE: This is an EXPERIMENTAL utility and exploits behavior which is not
          part of the FSDP public API. Use at your own risk.

    By moving optimizer step invocation into the backward call we can free
    gradients earlier and reduce peak memory.

    """
    from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
    from torch.distributed.fsdp._traversal_utils import _get_fsdp_handles
    from torch.distributed.fsdp.flat_param import FlatParameter, FlatParamHandle

    apply_lock = threading.Lock()

    param_handles = _get_fsdp_handles(module)
    assert param_handles, f"Module {module} does not appear to contain any FSDP modules."
    fsdp_state = _get_module_fsdp_state(module)
    assert fsdp_state is not None
    fsdp_stream = fsdp_state._post_backward_stream if _TORCH_GREATER_EQUAL_2_1 else fsdp_state._streams["post_backward"]

    if isinstance(optimizers, Optimizer):
        optimizers = [optimizers]

    # We cannot trigger the optimizer step until all parameters are ready.
    remaining = {}
    for optimizer in optimizers:
        unfinished: Dict[torch.nn.Parameter, None] = {}  # Use Dict as an ordered set.
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p not in unfinished:
                    assert p not in remaining, f"{p=} is shared between two optimizers."
                    unfinished[p] = None
                    remaining[p] = (optimizer, unfinished)

    def maybe_step(parameters: Iterable[torch.nn.Parameter], post_step: Callable[[], None] = _no_op) -> None:
        for p in tuple(parameters):
            optimizer, unfinished = remaining.pop(p)
            unfinished.pop(p)
            if not unfinished:
                optimizer.step()
                optimizer.zero_grad()

                # Used to call `_reset_flat_param_grad_info_if_needed`. Otherwise FSDP might hold on to the memory.
                post_step()

    try:
        hook_handles = []
        for h in param_handles:
            assert isinstance(h, FlatParamHandle)
            flat_param = h.flat_param
            fsdp_acc_grad, _ = flat_param._post_backward_hook_state  # type: ignore

            # We must take `h` and `flat_param` as arguments because Python
            # late binds closures.
            def _opt_hook(h: FlatParamHandle, flat_param: FlatParameter, *_unused: Any) -> None:
                assert flat_param._post_backward_called
                assert h.flat_param is flat_param
                with apply_lock, torch.cuda.stream(fsdp_stream):
                    # We invoke `prepare_gradient_for_optim` earlier than usual.
                    # We also need to prevent the later "normal" invocation,
                    # otherwise the double call will trigger FSDP asserts.
                    prepare_gradient = h.prepare_gradient_for_optim
                    assert hasattr(prepare_gradient, "__func__"), prepare_gradient
                    assert prepare_gradient.__func__ is FlatParamHandle.prepare_gradient_for_optim
                    prepare_gradient()
                    h.prepare_gradient_for_optim = _no_op  # type: ignore[method-assign]
                    post_step = (
                        h._reset_flat_param_grad_info_if_needed
                        if _TORCH_GREATER_EQUAL_2_1
                        else h._clear_grads_if_needed
                    )
                    maybe_step(flat_param._params or (), post_step=post_step)

            hook = partial(_opt_hook, h, flat_param)
            hook_handles.append(fsdp_acc_grad.register_hook(hook))

        yield

    finally:
        # Non-FSDP parameters won't have a grad hook, so handle them here.
        with apply_lock:
            maybe_step(remaining)

        # Unregister the grad hooks.
        for hook_handle in hook_handles:
            hook_handle.remove()

        # And lastly back out the handle monkey patches.
        for h in param_handles:
            if h.prepare_gradient_for_optim is _no_op:
                del h.prepare_gradient_for_optim


def fsdp_overlap_step_with_backward(
    optimizers: Union[Optimizer, Iterable[Optimizer]],
    fabric_module: "_FabricModule",
) -> Generator[None, None, None]:
    if not _TORCH_GREATER_EQUAL_2_0:
        raise NotImplementedError(
            "`fsdp_overlap_step_with_backward` requires torch >= 2.0.0. HINT: `pip install -U torch`"
        )

    from lightning.fabric.wrappers import _FabricModule

    assert isinstance(fabric_module, _FabricModule)
    return _apply_optimizers_during_fsdp_backward(  # type: ignore[return-value]
        optimizers, fabric_module._forward_module
    )
