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
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, Literal, Optional, TypeVar, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies.fsdp import (
    _distributed_checkpoint_load,
    _distributed_checkpoint_save,
    _has_meta_device_parameters_or_buffers,
)
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.strategy import TBroadcast, _BackwardSyncControl
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

TModel = TypeVar("TModel", bound=Module)


class ModelParallelStrategy(ParallelStrategy):
    """Enables user-defined parallelism applied to a model.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Currently supports up to 2D parallelism, for example Fully Sharded Data-Parallel combined with
    Tensor Parallelism. Requires PyTorch 2.3 or newer.

    Arguments:
        parallelize_fn: A function that applies parallelisms to a module. The strategy will provide the
            model and device mesh as input.
        data_parallel_size: The number of devices within a data-parallel group. Defaults to ``"auto"``, which
            sets this size to the number of nodes in the cluster.
        tensor_parallel_size: The number of devices within a tensor-parallel group. Defaults to ``"auto"``, which
            sets this size to the number of GPUs in a single node.

    """

    def __init__(
        self,
        parallelize_fn: Callable[[TModel, "DeviceMesh"], TModel],
        data_parallel_size: Union[Literal["auto"], int] = "auto",
        tensor_parallel_size: Union[Literal["auto"], int] = "auto",
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
    ) -> None:
        super().__init__()
        if not _TORCH_GREATER_EQUAL_2_3:
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.3 or higher.")
        self._parallelize_fn = parallelize_fn
        self._data_parallel_size = data_parallel_size
        self._tensor_parallel_size = tensor_parallel_size
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._backward_sync_control = _ParallelBackwardSyncControl()

        self._device_mesh: Optional["DeviceMesh"] = None

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
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
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
        self._setup_device_mesh()

    @override
    def setup_module(self, module: Module) -> Module:
        module = self._parallelize_fn(module, self.device_mesh)
        if not isinstance(module, Module):
            raise TypeError(
                f"The `parallelize_fn` must return a `nn.Module` instance, but got: {type(module).__name__}"
            )
        _materialize_module(module, self.root_device)
        return module

    @override
    def module_to_device(self, module: Module) -> None:
        pass

    @override
    def module_init_context(self, empty_init: Optional[bool] = None) -> ContextManager:
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
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state to a checkpoint on disk."""
        if storage_options is not None:
            raise TypeError(
                f"`{self.__class__.__name__}.save_checkpoint(..., storage_options=...)` is not supported because"
                f" `{self.__class__.__name__}` does not use the `CheckpointIO`."
            )
        if filter is not None:
            raise NotImplementedError(f"{self.__class__.__name__} does not yet support the `filter` argument.")

        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(path))
        _distributed_checkpoint_save(state, path)

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(state, (Module, Optimizer)):
            raise NotImplementedError(
                "Loading a module or optimizer object from a checkpoint directly is not yet supported."
            )
        if strict is False:
            raise NotImplementedError(f"Non-strict loading is not yet supported in {self.__class__.__name__}.")

        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(path))
        _distributed_checkpoint_load(state, path)  # type: ignore[arg-type]
        return {}

    def _setup_distributed(self) -> None:
        reset_seed()
        self._set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _setup_device_mesh(self) -> None:
        from torch.distributed.device_mesh import init_device_mesh

        if self._data_parallel_size == "auto":
            self._data_parallel_size = self.num_nodes
        if self._tensor_parallel_size == "auto":
            self._tensor_parallel_size = self.num_processes
        if self._data_parallel_size * self._tensor_parallel_size != self.world_size:
            raise RuntimeError(
                f"The sizes `data_parallel_size={self._data_parallel_size}` and"
                f" `tensor_parallel_size={self._tensor_parallel_size}` multiplied should equal the world size"
                f" ({self.world_size})."
            )
        self._device_mesh = init_device_mesh(
            device_type=self.root_device.type,
            mesh_shape=(self._data_parallel_size, self._tensor_parallel_size),
            mesh_dim_names=("data_parallel", "tensor_parallel"),
        )

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank


def _materialize_module(module: Module, device: torch.device) -> None:
    # Reference: https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md#meta-device-initialization
    # TODO: Introduce `Fabric.materialize(module)` to give user control when materialization should happen
    # TODO: Make `torchmetrics.Metric` compatible with the `to_empty()` + `reset_parameters()` semantics
    if not _has_meta_device_parameters_or_buffers(module):
        return

    module.to_empty(device=device)  # has to be called on the root module

    uninitialized_modules = set()
    for submodule in module.modules():
        if all(False for _ in itertools.chain(submodule.parameters(recurse=False), submodule.buffers(recurse=False))):
            # module has no parameters or buffers
            continue
        if callable(reset_method := getattr(submodule, "reset_parameters", None)):
            reset_method()
        else:
            uninitialized_modules.add(type(submodule).__name__)

    if uninitialized_modules:
        rank_zero_warn(
            "Parameter initialization incomplete. The following modules have parameters or buffers with uninitialized"
            " memory because they don't define a `reset_parameters()` method for re-initialization:"
            f" {', '.join(uninitialized_modules)}"
        )


class _ParallelBackwardSyncControl(_BackwardSyncControl):
    @override
    def no_backward_sync(self, module: Module, enabled: bool) -> ContextManager:
        """Blocks gradient synchronization inside the FSDP2 modules."""
        return _FSDPNoSync(module=module, enabled=enabled)


class _FSDPNoSync(ContextManager):
    def __init__(self, module: Module, enabled: bool) -> None:
        self._module = module
        self._enabled = enabled

    def _set_requires_grad_sync(self, requires_grad_sync: bool) -> None:
        from torch.distributed._composable.fsdp import FSDP

        for mod in self._module.modules():
            if isinstance(mod, FSDP):
                mod.set_requires_gradient_sync(requires_grad_sync, recurse=False)

    def __enter__(self) -> None:
        self._set_requires_grad_sync(not self._enabled)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._set_requires_grad_sync(self._enabled)
