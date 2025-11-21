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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

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
from lightning.fabric.strategies.strategy import TBroadcast
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import _PATH, ReduceOp

if TYPE_CHECKING:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader


class XLAStrategy(ParallelStrategy):
    """Strategy for training multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn`
    method."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        checkpoint_io: Optional[XLACheckpointIO] = None,
        precision: Optional[XLAPrecision] = None,
        sync_module_states: bool = True,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=XLAEnvironment(),
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.strategies.xla.ddp import XLAStrategyFabric as EnterpriseXLAStrategy

        self.xla_strategy_impl = EnterpriseXLAStrategy(outer_object=self, sync_module_states=sync_module_states)

    @property
    @override
    def root_device(self) -> torch.device:
        return self.xla_strategy_impl.root_device

    @property
    def num_processes(self) -> int:
        return self.xla_strategy_impl.num_processes

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
            raise TypeError(f"The XLA strategy can only work with the `XLAPrecision` plugin, found {precision}")
        self._precision = precision

    @property
    @override
    def global_rank(self) -> int:
        return self.xla_strategy_impl.global_rank

    @property
    @override
    def local_rank(self) -> int:
        return self.xla_strategy_impl.local_rank

    @property
    @override
    def node_rank(self) -> int:
        return self.xla_strategy_impl.node_rank

    @property
    @override
    def world_size(self) -> int:
        return self.xla_strategy_impl.world_size

    @override
    def _configure_launcher(self) -> None:
        self._launcher = _XLALauncher(self)

    @override
    def setup_environment(self) -> None:
        return self.xla_strategy_impl.setup_environment()

    @override
    def setup_module(self, module: Module) -> Module:
        return self.xla_strategy_impl.setup_module(module=module)

    @override
    def module_to_device(self, module: Module) -> None:
        return self.xla_strategy_impl.module_to_device(module=module)

    @override
    def process_dataloader(self, dataloader: DataLoader) -> "MpDeviceLoader":
        return self.xla_strategy_impl.process_dataloader(dataloader=dataloader)

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
        return self.xla_strategy_impl.all_gather(tensor=tensor, group=group, sync_grads=sync_grads)

    @override
    def all_reduce(
        self, output: Union[Tensor, Any], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> Tensor:
        return self.xla_strategy_impl.all_reduce(output=output, group=group, reduce_op=reduce_op)

    @override
    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        return self.xla_strategy_impl.barrier(name=name, *args, **kwargs)

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return self.xla_strategy_impl.broadcast(obj=obj, src=src)

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state as a checkpoint file.

        Args:
            path: A path to where the file(s) should be saved
            state: A dictionary with contents to be saved. If the dict contains modules or optimizers, their
                state-dict will be retrieved and converted automatically.
            storage_options: Additional options for the ``CheckpointIO`` plugin
            filter: An optional dictionary of the same format as ``state`` mapping keys to callables that return a
                boolean indicating whether the given parameter should be saved (``True``) or filtered out (``False``).

        """
        return self.xla_strategy_impl.save_checkpoint(
            path=path, state=state, storage_options=storage_options, filter=filter
        )

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("xla", cls, description=cls.__name__)
