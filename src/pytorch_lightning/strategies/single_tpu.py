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
import os
from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.plugins.io.xla_plugin import XLACheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities import _TPU_AVAILABLE, find_shared_parameters, set_shared_parameters

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


class SingleTPUStrategy(SingleDeviceStrategy):
    """Strategy for training on a single TPU device."""

    strategy_name = "single_tpu"

    def __init__(
        self,
        device: int,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
    ):
        super().__init__(
            accelerator=accelerator,
            device=xm.xla_device(device),
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self.debug = debug

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = XLACheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = XLACheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.model, "self.model must be set before find_shared_parameters(self.model)"
        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        set_shared_parameters(self.model, shared_params)
        super().setup(trainer)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = str(1)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def teardown(self) -> None:
        super().teardown()
        os.environ.pop("PT_XLA_DEBUG", None)
