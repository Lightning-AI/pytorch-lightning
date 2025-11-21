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
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins import CheckpointIO, Precision, XLAPrecision
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.plugins.io.xla import XLACheckpointIO
from lightning.fabric.strategies import ParallelStrategy, _StrategyRegistry
from lightning.fabric.strategies.launchers.xla import _XLALauncher
from lightning.fabric.strategies.strategy import (
    TBroadcast,
    _Sharded,
)
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import _PATH, Optimizable, ReduceOp

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import _LRScheduler
    from torch_xla.distributed.parallel_loader import MpDeviceLoader

_POLICY_SET = set[type[Module]]
_POLICY = Union[_POLICY_SET, Callable[[Module, bool, int], bool]]


class XLAFSDPStrategy(ParallelStrategy, _Sharded):
    r"""Strategy for training multiple XLA devices using the
    :func:`torch_xla.distributed.xla_fully_sharded_data_parallel.XlaFullyShardedDataParallel` method.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    For more information check out https://github.com/pytorch/xla/blob/v2.5.0/docs/fsdp.md

    Args:
        auto_wrap_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`.
            For convenience, this also accepts a set of the layer classes to wrap.
        activation_checkpointing_policy: Used when selecting the modules for
            which you want to enable activation checkpointing. Enabling this can free up a significant amount of memory
            at the cost of speed since activations in these layers need to be recomputed during backpropagation.
            This accepts a set of the layer classes to wrap.

        state_dict_type: The format in which the state of the model and optimizers gets saved into the checkpoint.

            - ``"full"``: The full weights and optimizer states get assembled on rank 0 and saved to a single file.
            - ``"sharded"``: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
              a folder with files for each shard in the host. Note that TPU VM multihost does not have a shared
              filesystem.

        sequential_save: With this enabled, individual ranks consecutively save their state dictionary shards, reducing
            peak system RAM usage, although it elongates the saving process.
        \**kwargs: See available parameters in :class:`torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`.

    """

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        checkpoint_io: Optional[XLACheckpointIO] = None,
        precision: Optional[XLAPrecision] = None,
        auto_wrap_policy: Optional[_POLICY] = None,
        activation_checkpointing_policy: Optional[_POLICY_SET] = None,
        state_dict_type: Literal["full", "sharded"] = "sharded",
        sequential_save: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=XLAEnvironment(),
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.strategies.xla.fsdp import (
            XLAFSDPStrategyFabric as EnterpriseXLAFSDPStrategy,
        )

        self.xla_fsdp_impl = EnterpriseXLAFSDPStrategy(
            outer_object=self,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy=activation_checkpointing_policy,
            state_dict_type=state_dict_type,
            sequential_save=sequential_save,
            **kwargs,
        )

    @property
    @override
    def root_device(self) -> torch.device:
        return self.xla_fsdp_impl.root_device

    @property
    def num_processes(self) -> int:
        return self.xla_fsdp_impl.num_processes

    @property
    @override
    def checkpoint_io(self) -> XLACheckpointIO:
        plugin = self._checkpoint_io
        if plugin is not None:
            assert isinstance(plugin, XLACheckpointIO)
            return plugin
        return XLACheckpointIO()

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        if io is not None and not isinstance(io, XLACheckpointIO):
            raise TypeError(f"The XLA strategy can only work with the `XLACheckpointIO` plugin, found {io}")
        self._checkpoint_io = io

    @property
    @override
    def precision(self) -> XLAPrecision:
        plugin = self._precision
        if plugin is not None:
            assert isinstance(plugin, XLAPrecision)
            return plugin
        return XLAPrecision("32-true")

    @precision.setter
    @override
    def precision(self, precision: Optional[Precision]) -> None:
        if precision is not None and not isinstance(precision, XLAPrecision):
            raise TypeError(f"The XLA FSDP strategy can only work with the `XLAPrecision` plugin, found {precision}")
        self._precision = precision

    @property
    @override
    def global_rank(self) -> int:
        return self.xla_fsdp_impl.global_rank

    @property
    @override
    def local_rank(self) -> int:
        return self.xla_fsdp_impl.local_rank

    @property
    @override
    def node_rank(self) -> int:
        return self.xla_fsdp_impl.node_rank

    @property
    @override
    def world_size(self) -> int:
        return self.xla_fsdp_impl.world_size

    @override
    def _configure_launcher(self) -> None:
        self._launcher = _XLALauncher(self)

    @override
    def setup_environment(self) -> None:
        return self.xla_fsdp_impl.setup_environment()

    @override
    def setup_module_and_optimizers(
        self, module: Module, optimizers: list[Optimizer], scheduler: Optional["_LRScheduler"] = None
    ) -> tuple[Module, list[Optimizer], Optional["_LRScheduler"]]:
        return self.xla_fsdp_impl.setup_module_and_optimizers(module=module, optimizers=optimizers, scheduler=scheduler)

    @override
    def setup_module(self, module: Module) -> Module:
        return self.xla_fsdp_impl.setup_module(module=module)

    @override
    def module_to_device(self, module: Module) -> None:
        return self.xla_fsdp_impl.module_to_device(module=module)

    def module_init_context(self, empty_init: Optional[bool] = None) -> AbstractContextManager:
        return self.xla_fsdp_impl.module_init_context(empty_init=empty_init)

    @override
    def module_sharded_context(self) -> AbstractContextManager:
        return self.xla_fsdp_impl.module_sharded_context()

    @override
    def process_dataloader(self, dataloader: DataLoader) -> "MpDeviceLoader":
        return self.xla_fsdp_impl.process_dataloader(dataloader=dataloader)

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        return self.xla_fsdp_impl.setup_optimizer(optimizer=optimizer)

    @override
    def optimizer_step(self, optimizer: Optimizable, **kwargs: Any) -> Any:
        return self.xla_fsdp_impl.optimizer_step(optimizer=optimizer, **kwargs)

    @override
    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        """Clip gradients by norm."""
        return self.xla_fsdp_impl.clip_gradients_norm(
            module=module,
            optimizer=optimizer,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
        )

    @override
    def clip_gradients_value(self, module: Module, optimizer: Optimizer, clip_val: Union[float, int]) -> None:
        """Clip gradients by value."""
        return self.xla_fsdp_impl.clip_gradients_value(module=module, optimizer=optimizer, clip_val=clip_val)

    @override
    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Function to gather a tensor from several distributed processes.

        Args:
            tensor: tensor to all-gather.
            group: unused.
            sync_grads: flag that allows users to synchronize gradients for the all-gather operation.
        Return:
            A tensor of shape (world_size, ...)

        """
        return self.xla_fsdp_impl.all_gather(tensor=tensor, group=group, sync_grads=sync_grads)

    @override
    def all_reduce(
        self, output: Union[Tensor, Any], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> Tensor:
        return self.xla_fsdp_impl.all_reduce(output=output, group=group, reduce_op=reduce_op)

    @override
    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        return self.xla_fsdp_impl.barrier(name=name, *args, **kwargs)

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return self.xla_fsdp_impl.broadcast(obj=obj, src=src)

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state in the provided checkpoint directory.

        If the user specifies sharded checkpointing, the directory will contain one file per process, with model- and
        optimizer shards stored per file. If the user specifies full checkpointing, the directory will contain a
        consolidated checkpoint combining all of the sharded checkpoints.

        """
        return self.xla_fsdp_impl.save_checkpoint(
            path=path, state=state, storage_options=storage_options, filter=filter
        )

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
        weights_only: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Given a folder, load the contents from a checkpoint and restore the state of the given objects.

        The strategy currently only supports saving and loading sharded checkpoints which are stored in form of a
        directory of multiple files rather than a single file.

        """
        return self.xla_fsdp_impl.load_checkpoint(path=path, state=state, strict=strict, weights_only=weights_only)

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("xla_fsdp", cls, description=cls.__name__)
