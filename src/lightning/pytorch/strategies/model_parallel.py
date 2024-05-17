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
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Mapping, Optional, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies.model_parallel import _setup_device_mesh
from lightning.fabric.utilities.distributed import (
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.fabric.utilities.init import _materialize_distributed_module
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, ReduceOp
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class ModelParallelStrategy(ParallelStrategy):
    """Enables user-defined parallelism applied to a model.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Currently supports up to 2D parallelism. Specifically, it supports the combination of
    Fully Sharded Data-Parallel 2 (FSDP2) with Tensor Parallelism (DTensor). These PyTorch APIs are currently still
    experimental in PyTorch. Requires PyTorch 2.3 or newer.

    Arguments:
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
        data_parallel_size: Union[Literal["auto"], int] = "auto",
        tensor_parallel_size: Union[Literal["auto"], int] = "auto",
        save_distributed_checkpoint: bool = True,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
    ) -> None:
        super().__init__()
        if not _TORCH_GREATER_EQUAL_2_3:
            raise ImportError(f"{type(self).__name__} requires PyTorch 2.3 or higher.")
        self._data_parallel_size = data_parallel_size
        self._tensor_parallel_size = tensor_parallel_size
        self._save_distributed_checkpoint = save_distributed_checkpoint
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._device_mesh: Optional["DeviceMesh"] = None
        self.num_nodes = 1

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

    @property
    @override
    def restore_checkpoint_after_setup(self) -> bool:
        return True

    @property
    @override
    def lightning_restore_optimizer(self) -> bool:
        return False

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
        # Users can access device mesh in `LightningModule.configure_model()`
        self.lightning_module._device_mesh = self._device_mesh

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        from torch.distributed.fsdp import FullyShardedDataParallel

        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        if not is_overridden("configure_model", self.lightning_module):
            raise TypeError(
                f"When using the {type(self).__name__}, you are required to override the `configure_model()` hook in"
                f" the LightningModule and apply parallelization there."
            )
        if any(isinstance(mod, FullyShardedDataParallel) for mod in self.model.modules()):
            raise TypeError(
                "Found modules that are wrapped with `torch.distributed.fsdp.FullyShardedDataParallel`."
                f" The `{self.__class__.__name__}` only supports the new FSDP2 APIs in PyTorch >= 2.3."
            )

        _materialize_distributed_module(self.model, self.root_device)

        self.model = self.precision_plugin.convert_module(self.model)
        self.model_to_device()  # move all remaining layers if any left on CPU.

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

        return super().setup_optimizers(trainer)

    @override
    def model_to_device(self) -> None:
        assert self.model is not None
        self.model.to(self.root_device)

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        # Materializaton happens in `setup()`
        empty_init_context = torch.device("meta") if empty_init else nullcontext()
        with empty_init_context, self.precision_plugin.tensor_init_context():
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
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def _determine_device_ids(self) -> List[int]:
        return [self.root_device.index]

    @override
    def teardown(self) -> None:
        assert self.cluster_environment is not None
        assert self.accelerator is not None
        self.cluster_environment.teardown()
        self.precision_plugin.teardown()
        self.accelerator.teardown()

    @override
    def lightning_module_state_dict(self) -> Dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state_dict_options = StateDictOptions(full_state_dict=(not self._save_distributed_checkpoint), cpu_offload=True)
        assert self.model is not None
        return get_model_state_dict(self.model, options=state_dict_options)

    @override
    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        # Override to do nothing, the strategy already loaded the states in `load_checkpoint()`
        pass

    @override
    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import OptimStateKeyType

        state_dict_options = StateDictOptions(full_state_dict=(not self._save_distributed_checkpoint), cpu_offload=True)
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer

        assert self.model is not None
        state_dict = get_optimizer_state_dict(self.model, optimizer, options=state_dict_options)
        if not self._save_distributed_checkpoint:
            # Store the optimizer state dict in standard format
            state_dict = FSDP.rekey_optim_state_dict(state_dict, OptimStateKeyType.PARAM_ID, self.model)
        return state_dict

    @override
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Override to do nothing, the strategy already loaded the states in `load_checkpoint()`
        pass

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                f"`{type(self).__name__}.save_checkpoint(..., storage_options=...)` is not supported because"
                f" `{type(self).__name__}` does not use the `CheckpointIO`."
            )
        raise NotImplementedError("Checkpoint saving is not yet implemented.")

    @override
    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        raise NotImplementedError("Checkpoint loading is not yet implemented.")

    def _setup_distributed(self) -> None:
        super().setup_environment()
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank
