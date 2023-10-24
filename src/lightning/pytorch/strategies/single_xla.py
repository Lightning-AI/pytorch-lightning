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
from typing import Optional, Union

import torch

import lightning.pytorch as pl
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins import XLACheckpointIO
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from lightning.pytorch.utilities import find_shared_parameters, set_shared_parameters


class SingleDeviceXLAStrategy(SingleDeviceStrategy):
    """Strategy for training on a single XLA device."""

    def __init__(
        self,
        device: _DEVICE,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        checkpoint_io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]] = None,
        precision_plugin: Optional[XLAPrecisionPlugin] = None,
        debug: bool = False,
    ):
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        if isinstance(device, torch.device):
            # unwrap the `torch.device` in favor of `xla_device`
            device = device.index
        import torch_xla.core.xla_model as xm

        super().__init__(
            accelerator=accelerator,
            device=xm.xla_device(device),
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self.debug = debug

    @property  # type: ignore[override]
    def checkpoint_io(self) -> Union[XLACheckpointIO, _WrappingCheckpointIO]:
        plugin = self._checkpoint_io
        if plugin is not None:
            assert isinstance(plugin, (XLACheckpointIO, _WrappingCheckpointIO))
            return plugin
        return XLACheckpointIO()

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]]) -> None:
        if io is not None and not isinstance(io, (XLACheckpointIO, _WrappingCheckpointIO)):
            raise TypeError(f"The XLA strategy can only work with the `XLACheckpointIO` plugin, found {io}")
        self._checkpoint_io = io

    @property  # type: ignore[override]
    def precision_plugin(self) -> XLAPrecisionPlugin:
        plugin = self._precision_plugin
        if plugin is not None:
            assert isinstance(plugin, XLAPrecisionPlugin)
            return plugin
        return XLAPrecisionPlugin()

    @precision_plugin.setter
    def precision_plugin(self, precision_plugin: Optional[XLAPrecisionPlugin]) -> None:
        if precision_plugin is not None and not isinstance(precision_plugin, XLAPrecisionPlugin):
            raise TypeError(
                f"The XLA strategy can only work with the `XLAPrecisionPlugin` plugin, found {precision_plugin}"
            )
        self._precision_plugin = precision_plugin

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.model, "self.model must be set before find_shared_parameters(self.model)"
        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        set_shared_parameters(self.model, shared_params)
        super().setup(trainer)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = str(1)

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("single_xla", cls, description=cls.__name__)

    def teardown(self) -> None:
        super().teardown()
        os.environ.pop("PT_XLA_DEBUG", None)
