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
import itertools
import shutil
from collections.abc import Generator
from contextlib import AbstractContextManager, ExitStack
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypeVar, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import TypeGuard, override

from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies.fsdp import (
    _distributed_checkpoint_load,
    _distributed_checkpoint_save,
    _get_full_state_dict_context,
    _is_full_checkpoint,
    _is_sharded_checkpoint,
)
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.strategy import (
    TBroadcast,
    _apply_filter,
    _BackwardSyncControl,
    _validate_keys_for_strict_loading,
)
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3, _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.utilities.init import _materialize_distributed_module
from lightning.fabric.utilities.load import _METADATA_FILENAME, _lazy_load, _move_state_into
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, _Stateful

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

TModel = TypeVar("TModel", bound=Module)


class ModelParallelStrategy(ParallelStrategy):
    """Enables user-defined parallelism applied to a model.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Currently supports up to 2D parallelism. Specifically, it supports the combination of
    Fully Sharded Data-Parallel 2 (FSDP2) with Tensor Parallelism (DTensor). These PyTorch APIs are currently still
    experimental in PyTorch. Requires PyTorch 2.4 or newer.

    Arguments:
        parallelize_fn: A function that applies parallelisms to a module. The strategy will provide the
            model and device mesh as input.
        data_parallel_size: The number of devices within a data-parallel group. Defaults to ``"auto"``, which
            sets this size to the number of nodes in the cluster.
        tensor_parallel_size: The number of devices within a tensor-parallel group. Defaults to ``"auto"``, which
            sets this size to the number of GPUs in a single node.
        save_distributed_checkpoint: If ``True``, each rank saves its shard of weights and optimizer states to a file.
            The checkpoint is a folder with as many files as the world size.
            If ``False``, the full weights and optimizer states get assembled on rank 0 and saved to a single file.

    """

    def __init__(
        self,
        parallelize_fn: Callable[[TModel, "DeviceMesh"], TModel],
        data_parallel_size: Union[Literal["auto"], int] = "auto",
        tensor_parallel_size: Union[Literal["auto"], int] = "auto",
        save_distributed_checkpoint: bool = True,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
    ) -> None:
        super().__init__()
        if not _TORCH_GREATER_EQUAL_2_4:
            raise ImportError(f"{type(self).__name__} requires PyTorch 2.4 or higher.")
        self._parallelize_fn = parallelize_fn
        self._data_parallel_size = data_parallel_size
        self._tensor_parallel_size = tensor_parallel_size
        self._num_nodes = 1
        self._save_distributed_checkpoint = save_distributed_checkpoint
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._backward_sync_control = _ParallelBackwardSyncControl()

        self._device_mesh: Optional[DeviceMesh] = None

    @property
    def device_mesh(self) -> "DeviceMesh":
        if self._device_mesh is None:
            raise RuntimeError("Accessing the device mesh before processes have initialized is not allowed.")
        return self._device_mesh

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        raise NotImplementedError(f"The `{type(self).__name__}` does not use the `CheckpointIO` plugin interface.")

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: CheckpointIO) -> None:
        raise NotImplementedError(f"The `{type(self).__name__}` does not support setting a `CheckpointIO` plugin.")

    @property
    @override
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
    @override
    def distributed_sampler_kwargs(self) -> dict[str, Any]:
        assert self.device_mesh is not None
        data_parallel_mesh = self.device_mesh["data_parallel"]
        return {"num_replicas": data_parallel_mesh.size(), "rank": data_parallel_mesh.get_local_rank()}

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()
        if self._data_parallel_size == "auto":
            self._data_parallel_size = self.num_nodes
        if self._tensor_parallel_size == "auto":
            self._tensor_parallel_size = self.num_processes
        self._device_mesh = _setup_device_mesh(
            self._data_parallel_size, self._tensor_parallel_size, self.world_size, self.root_device
        )

    @override
    def setup_module(self, module: Module) -> Module:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if any(isinstance(mod, FullyShardedDataParallel) for mod in module.modules()):
            raise TypeError(
                "Found modules that are wrapped with `torch.distributed.fsdp.FullyShardedDataParallel`."
                f" The `{self.__class__.__name__}` only supports the new FSDP2 APIs in PyTorch >= 2.4."
            )

        module = self._parallelize_fn(module, self.device_mesh)  # type: ignore[arg-type]
        if not isinstance(module, Module):
            raise TypeError(
                f"The `parallelize_fn` must return a `nn.Module` instance, but got: {type(module).__name__}"
            )
        _materialize_distributed_module(module, self.root_device)
        return module

    @override
    def module_to_device(self, module: Module) -> None:
        pass

    @override
    def module_init_context(self, empty_init: Optional[bool] = None) -> AbstractContextManager:
        precision_init_ctx = self.precision.module_init_context()
        stack = ExitStack()
        if empty_init:
            # Materializaton happens in `setup_module`
            # TODO: Introduce `Fabric.materialize(module)` to give user control over materialization
            stack.enter_context(torch.device("meta"))
        stack.enter_context(precision_init_ctx)
        return stack

    @override
    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[self.root_device.index])
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
    def save_checkpoint(
        self,
        path: _PATH,
        state: dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state to a checkpoint on disk.

        If distributed checkpointing is enabled (default), the checkpoint gets saved as a directory containing one file
        per process, with model- and optimizer shards stored per file. Additionally, it creates a metadata file
        `meta.pt` with the rest of the user's state (only saved from rank 0).
        If distributed checkpointing is disabled (``save_distributed_checkpoint=False``), the checkpoint will be
        written to a single file containing the weights, optimizer state and other metadata.

        """
        if storage_options is not None:
            raise TypeError(
                f"`{type(self).__name__}.save_checkpoint(..., storage_options=...)` is not supported because"
                f" `{type(self).__name__}` does not use the `CheckpointIO`."
            )
        if filter is not None and self._save_distributed_checkpoint:
            # https://github.com/pytorch/pytorch/issues/105379
            raise NotImplementedError(
                f"{type(self).__name__} doesn't support loading distributed filtered checkpoints,"
                " so saving them is disabled."
            )
        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(path))
        _save_checkpoint(
            path=path,
            state=state,
            full_state_dict=(not self._save_distributed_checkpoint),
            rank=self.global_rank,
            filter=filter,
        )

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load the contents from a checkpoint and restore the state of the given objects."""
        if not state:
            raise ValueError(
                f"Got {type(self).__name__}.load_checkpoint(..., state={state!r}) but a state with at least "
                " a model instance to reload is required. Pass it in like so:"
                f" {type(self).__name__}.load_checkpoint(..., state={{'model': model, ...}})"
            )
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(path))

        if isinstance(state, Module):
            _load_raw_module_state_from_path(path, module=state, world_size=self.world_size, strict=strict)
            return {}

        if isinstance(state, Optimizer):
            raise NotImplementedError(
                f"Loading a single optimizer object from a checkpoint is not supported yet with {type(self).__name__}."
            )

        return _load_checkpoint(path=path, state=state, strict=strict)

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
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank


class _ParallelBackwardSyncControl(_BackwardSyncControl):
    @override
    def no_backward_sync(self, module: Module, enabled: bool) -> AbstractContextManager:
        """Blocks gradient synchronization inside the FSDP2 modules."""
        return _FSDPNoSync(module=module, enabled=enabled)


class _FSDPNoSync(AbstractContextManager):
    def __init__(self, module: Module, enabled: bool) -> None:
        self._module = module
        self._enabled = enabled

    def _set_requires_grad_sync(self, requires_grad_sync: bool) -> None:
        from torch.distributed._composable.fsdp import FSDPModule

        for mod in self._module.modules():
            if isinstance(mod, FSDPModule):
                mod.set_requires_gradient_sync(requires_grad_sync, recurse=False)

    def __enter__(self) -> None:
        self._set_requires_grad_sync(not self._enabled)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._set_requires_grad_sync(self._enabled)


def _save_checkpoint(
    path: Path,
    state: dict[str, Union[Module, Optimizer, Any]],
    full_state_dict: bool,
    rank: int,
    filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
) -> None:
    if path.is_dir() and full_state_dict and not _is_sharded_checkpoint(path):
        raise IsADirectoryError(f"The checkpoint path exists and is a directory: {path}")

    modules = [module for module in state.values() if _has_dtensor_modules(module)]
    if len(modules) == 0:
        raise ValueError(
            "Could not find a distributed model in the provided checkpoint state. Please provide the model as"
            " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
            " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
        )
    if len(modules) > 1:
        raise ValueError(
            "Found multiple distributed models in the given state. Saving distributed checkpoints is"
            " currently limited to a single model per checkpoint. To save multiple models, call the"
            " save method for each model separately with a different path."
        )
    module = modules[0]

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_optimizer_state_dict

    state_dict_options = StateDictOptions(full_state_dict=full_state_dict, cpu_offload=True)

    # replace the modules and optimizer objects in the state with their local state dict
    # and separate the user's metadata
    converted_state: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for key, obj in state.items():
        converted: Any
        if isinstance(obj, Module):
            converted = get_model_state_dict(obj, options=state_dict_options)
            target_dict = converted_state
        elif isinstance(obj, Optimizer):
            converted = get_optimizer_state_dict(module, obj, options=state_dict_options)
            target_dict = converted_state
        else:  # everything not a module or optimizer is considered metadata
            converted = obj.state_dict() if isinstance(obj, _Stateful) else obj
            target_dict = metadata
        _apply_filter(key, filter or {}, converted, target_dict)

    if full_state_dict:
        if _is_sharded_checkpoint(path):
            shutil.rmtree(path)
        converted_state.update(metadata)
        if rank == 0:
            torch.save(converted_state, path)
    else:
        if path.is_file():
            path.unlink()
        path.mkdir(parents=True, exist_ok=True)
        _distributed_checkpoint_save(converted_state, path)
        if rank == 0:
            torch.save(metadata, path / _METADATA_FILENAME)


def _load_checkpoint(
    path: Path,
    state: dict[str, Union[Module, Optimizer, Any]],
    strict: bool = True,
    optimizer_states_from_list: bool = False,
) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    modules = {key: module for key, module in state.items() if _has_dtensor_modules(module)}
    if len(modules) == 0:
        raise ValueError(
            "Could not find a distributed model in the provided checkpoint state. Please provide the model as"
            " part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure"
            " you set up the model (and optimizers if any) through the strategy before loading the checkpoint."
        )
    optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
    if len(modules) > 1:
        raise ValueError(
            "Found multiple distributed models in the given state. Loading distributed checkpoints is"
            " currently limited to a single model per checkpoint. To load multiple models, call the"
            " load method for each model separately with a different path."
        )
    module_key, module = list(modules.items())[0]

    if _is_sharded_checkpoint(path):
        state_dict_options = StateDictOptions(cpu_offload=True)

        module_state = {module_key: get_model_state_dict(module)}
        _distributed_checkpoint_load(module_state, path)
        module.load_state_dict(module_state[module_key], strict=strict)

        # the optimizer states must be loaded separately
        for optim_key, optim in optimizers.items():
            optim_state = {optim_key: get_optimizer_state_dict(module, optim)}
            _distributed_checkpoint_load(optim_state, path)
            set_optimizer_state_dict(module, optim, optim_state_dict=optim_state[optim_key], options=state_dict_options)

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
        checkpoint = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
        _load_raw_module_state(checkpoint.pop(module_key), module, strict=strict)

        state_dict_options = StateDictOptions(
            broadcast_from_rank0=True,
            full_state_dict=True,
            strict=strict,
        )
        for optimizer_idx, (optimizer_name, optimizer) in enumerate(optimizers.items()):
            if optimizer_states_from_list:
                # This code path is only used by `lightning.pytorch`, which saves optimizer states as a list
                # rather than individual states at the top level.
                optimizer_state = checkpoint["optimizer_states"][optimizer_idx]
            else:
                optimizer_state = checkpoint.pop(optimizer_name)

            optimizer_state = _rekey_optimizer_state_if_needed(optimizer_state, module)
            set_optimizer_state_dict(
                module,
                optimizer,
                optim_state_dict=optimizer_state,
                options=state_dict_options,
            )

        requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
        _validate_keys_for_strict_loading(requested_metadata_keys, checkpoint.keys(), strict=strict)

        # Load metadata (anything not a module or optimizer)
        _move_state_into(source=checkpoint, destination=state, keys=requested_metadata_keys)

        # return the remaining metadata that wasn't requested as part of `state`
        return checkpoint

    raise ValueError(
        f"The path {str(path)!r} does not point to a valid checkpoint. Make sure the path points to either a"
        " directory with distributed checkpoint shards, or a single file with a full checkpoint."
    )


def _setup_device_mesh(
    data_parallel_size: int,
    tensor_parallel_size: int,
    world_size: int,
    device: torch.device,
) -> "DeviceMesh":
    from torch.distributed.device_mesh import init_device_mesh

    if data_parallel_size * tensor_parallel_size != world_size:
        raise RuntimeError(
            f"The sizes `data_parallel_size={data_parallel_size}` and"
            f" `tensor_parallel_size={tensor_parallel_size}` multiplied should equal the world size"
            f" ({world_size})."
        )
    return init_device_mesh(
        device_type=device.type,
        mesh_shape=(data_parallel_size, tensor_parallel_size),
        mesh_dim_names=("data_parallel", "tensor_parallel"),
    )


def _has_dtensor_modules(module: object) -> TypeGuard[Module]:
    from torch.distributed._tensor import DTensor

    return isinstance(module, Module) and any(isinstance(t, DTensor) for t in module.parameters())


def _load_raw_module_state_from_path(path: Path, module: Module, world_size: int, strict: bool = True) -> None:
    """Loads the state dict from a file path into the FSDP module."""
    if not _is_full_checkpoint(path):
        raise ValueError(
            "Failed to load checkpoint directly into the model. The given path must be a single file containing the"
            f" full state dict: {path}"
        )
    # Use `lazy_load`/`mmap` instead to avoid storing a copy of the full checkpoint per rank
    state_dict = torch.load(path, mmap=True, map_location="cpu") if _TORCH_GREATER_EQUAL_2_3 else _lazy_load(path)
    _load_raw_module_state(state_dict=state_dict, module=module, world_size=world_size, strict=strict)


def _load_raw_module_state(
    state_dict: dict[str, Any], module: Module, world_size: int = 1, strict: bool = True
) -> None:
    """Loads the state dict into the module by gathering all weights first and then and writing back to each shard."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if _has_dtensor_modules(module):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        state_dict_options = StateDictOptions(
            broadcast_from_rank0=True,
            full_state_dict=True,
            # must be set False to allow loading each param separately below
            strict=False,
        )

        for submodule_name, submodule in module.named_modules():
            for param_name, _ in _named_parameters_and_buffers_to_load(submodule):
                full_param_name = f"{submodule_name}{'.' if submodule_name else ''}{param_name}"
                if full_param_name not in state_dict:
                    if not strict:
                        continue
                    raise KeyError(
                        f"The model contains a key '{full_param_name}' that does not exist in the loaded checkpoint."
                        " To disable strict loading, set `strict=False`."
                    )
                local_state_dict = {param_name: state_dict[full_param_name]}
                set_model_state_dict(submodule, local_state_dict, options=state_dict_options)

    elif isinstance(module, FSDP):
        with _get_full_state_dict_context(module, world_size=world_size, rank0_only=False):
            module.load_state_dict(state_dict, strict=strict)
    else:
        module.load_state_dict(state_dict, strict=strict)


def _named_parameters_and_buffers_to_load(module: Module) -> Generator:
    """Returns parameters and buffers, with non-persistent buffers excluded."""
    for param_name, param in itertools.chain(
        module.named_buffers(recurse=False),
        module.named_parameters(recurse=False),
    ):
        if param_name in module._non_persistent_buffers_set:
            continue
        yield param_name, param


def _rekey_optimizer_state_if_needed(optimizer_state_dict: dict[str, Any], module: Module) -> dict[str, Any]:
    """Handles the case where the optimizer state is saved from a normal optimizer and converts the keys to parameter
    names."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import OptimStateKeyType

    if isinstance(list(optimizer_state_dict["state"].keys())[0], int):
        optimizer_state_dict = FSDP.rekey_optim_state_dict(optimizer_state_dict, OptimStateKeyType.PARAM_NAME, module)
    return optimizer_state_dict
