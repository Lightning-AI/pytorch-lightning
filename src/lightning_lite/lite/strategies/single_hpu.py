# Copyright The PyTorch Lightning team.
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

from typing import Dict, Optional

from lightning_lite.lite.plugins.io.checkpoint_plugin import CheckpointIO
from lightning_lite.lite.plugins.io.hpu_plugin import HPUCheckpointIO
from lightning_lite.lite.plugins.io.wrapper import _WrappingCheckpointIO
from lightning_lite.lite.plugins.precision import PrecisionPlugin
from lightning_lite.lite.strategies.single_device import SingleDeviceStrategy
from lightning_lite.lite.utilities import _HPU_AVAILABLE
from lightning_lite.lite.utilities.types import _DEVICE



class SingleHPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single HPU device."""

    strategy_name = "hpu_single"

    def __init__(
        self,
        device: _DEVICE = "hpu",
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):

        if not _HPU_AVAILABLE:
            raise RuntimeError("`SingleHPUStrategy` requires HPU devices to run")

        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = HPUCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = HPUCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        super().setup_optimizers(trainer)

    def model_to_device(self) -> None:
        self.model.to(self.root_device)  # type: ignore

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
