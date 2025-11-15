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
from typing import Optional, Union

from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO, Precision, XLACheckpointIO
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.plugins.precision.xla import XLAPrecision
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy


class SingleDeviceXLAStrategy(SingleDeviceStrategy):
    """Strategy for training on a single XLA device."""

    def __init__(
        self,
        device: _DEVICE,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        checkpoint_io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]] = None,
        precision_plugin: Optional[XLAPrecision] = None,
        debug: bool = False,
    ):
        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.strategies.xla.single import (
            SingleDeviceXLAStrategyTrainer as EnterpriseSingleDeviceXLAStrategy,
        )

        self.single_xla_strategy_impl = EnterpriseSingleDeviceXLAStrategy(outer_object=self, device=device, debug=debug)

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

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        return self.single_xla_strategy_impl.setup(trainer=trainer)

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("single_xla", cls, description=cls.__name__)

    @override
    def teardown(self) -> None:
        return self.single_xla_strategy_impl.teardown()
