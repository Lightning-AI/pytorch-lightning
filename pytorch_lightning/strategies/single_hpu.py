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

import pytorch_lightning as pl
from pytorch_lightning.plugins.io.hpu_plugin import HPUCheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _DEVICE

if _HPU_AVAILABLE:
    import habana_frameworks.torch.core.hccl  # noqa: F401
    from habana_frameworks.torch.utils.library_loader import load_habana_module


class SingleHPUStrategy(SingleDeviceStrategy):
    """Strategy for training on single HPU device."""

    strategy_name = "hpu_single"

    def __init__(
        self,
        device: _DEVICE = "hpu",
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        checkpoint_io: Optional[HPUCheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):

        if not _HPU_AVAILABLE:
            raise MisconfigurationException("`SingleHPUStrategy` requires HPU devices to run")

        # This function is used to load Habana libraries required for PyTorch
        # to register HPU as one of the available devices.
        load_habana_module()

        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io or HPUCheckpointIO(),
            precision_plugin=precision_plugin,
        )

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        super().setup_optimizers(trainer)

        if len(self.optimizers) > 1:
            raise MisconfigurationException("HPUs currently support only one optimizer.")

    def model_to_device(self) -> None:
        self.model.to(self.root_device)  # type: ignore

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
