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
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO, Precision, XLACheckpointIO
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import _PATH, ReduceOp
from lightning.pytorch.plugins import XLAPrecision
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.launchers.xla import _XLALauncher
from lightning.pytorch.strategies.strategy import TBroadcast

if TYPE_CHECKING:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader


class XLAStrategy(DDPStrategy):
    """Strategy for training multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn`
    method."""

    strategy_name = "xla"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        checkpoint_io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]] = None,
        precision_plugin: Optional[XLAPrecision] = None,
        debug: bool = False,
        sync_module_states: bool = True,
        **_: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=XLAEnvironment(),
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            start_method="fork",
        )
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.strategies.xla.ddp import XLAStrategyTrainer as EnterpriseXLAStrategy

        self.xla_strategy_impl = EnterpriseXLAStrategy(
            outer_object=self, debug=debug, sync_module_states=sync_module_states
        )

    @property
    @override
    def checkpoint_io(self) -> Union[XLACheckpointIO, _WrappingCheckpointIO]:
        plugin = self._checkpoint_io
        if plugin is not None:
            assert isinstance(plugin, (XLACheckpointIO, _WrappingCheckpointIO))
            return plugin
        return XLACheckpointIO()

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        if io is not None and not isinstance(io, (XLACheckpointIO, _WrappingCheckpointIO)):
            raise TypeError(f"The XLA strategy can only work with the `XLACheckpointIO` plugin, found {io}")
        self._checkpoint_io = io

    @property
    @override
    def precision_plugin(self) -> XLAPrecision:
        plugin = self._precision_plugin
        if plugin is not None:
            assert isinstance(plugin, XLAPrecision)
            return plugin
        return XLAPrecision()

    @precision_plugin.setter
    @override
    def precision_plugin(self, precision_plugin: Optional[Precision]) -> None:
        if precision_plugin is not None and not isinstance(precision_plugin, XLAPrecision):
            raise TypeError(f"The XLA strategy can only work with the `XLAPrecision` plugin, found {precision_plugin}")
        self._precision_plugin = precision_plugin

    @property
    @override
    def root_device(self) -> torch.device:
        return self.xla_strategy_impl.root_device

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
    def setup(self, trainer: "pl.Trainer") -> None:
        return self.xla_strategy_impl.setup(trainer=trainer)

    @override
    def _setup_model(self, model: Module) -> Module:  # type: ignore
        return self.xla_strategy_impl._setup_model(model=model)

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int]:
        return self.xla_strategy_impl.distributed_sampler_kwargs

    @override
    def process_dataloader(self, dataloader: object) -> "MpDeviceLoader":
        return self.xla_strategy_impl.process_dataloader(dataloader=dataloader)

    @override
    def configure_ddp(self) -> None:
        return self.xla_strategy_impl.configure_ddp()

    @override
    def model_to_device(self) -> None:
        return self.xla_strategy_impl.model_to_device()

    @override
    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        return self.xla_strategy_impl.barrier(name=name, *args, **kwargs)

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return self.xla_strategy_impl.broadcast(obj=obj, src=src)

    @override
    def reduce(
        self,
        output: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Tensor:
        return self.xla_strategy_impl.reduce(output=output, group=group, reduce_op=reduce_op)

    @override
    def setup_environment(self) -> None:
        return self.xla_strategy_impl.setup_environment()

    @override
    def setup_distributed(self) -> None:
        return self.xla_strategy_impl.setup_distributed()

    @override
    def set_world_ranks(self) -> None:
        return self.xla_strategy_impl.set_world_ranks()

    @override
    def save_checkpoint(
        self, checkpoint: dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        return self.xla_strategy_impl.save_checkpoint(
            checkpoint=checkpoint, filepath=filepath, storage_options=storage_options
        )

    @override
    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint

        """
        return self.xla_strategy_impl.remove_checkpoint(filepath=filepath)

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
    def teardown(self) -> None:
        return self.xla_strategy_impl.teardown()

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("xla_debug", cls, description="XLA strategy with `debug` as True", debug=True)
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=cls.__name__,
        )
