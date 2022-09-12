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

from lightning_lite.accelerators import Accelerator
from lightning_lite.plugins.io.checkpoint_plugin import CheckpointIO
from lightning_lite.plugins.io.xla_plugin import XLACheckpointIO
from lightning_lite.plugins.precision import Precision
from lightning_lite.strategies.single_device import SingleDeviceStrategy


class SingleTPUStrategy(SingleDeviceStrategy):
    """Strategy for training on a single TPU device."""

    strategy_name = "single_tpu"

    def __init__(
        self,
        device: int,
        accelerator: Optional[Accelerator] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[Precision] = None,
        debug: bool = False,
    ):
        import torch_xla.core.xla_model as xm

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
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def is_distributed(self) -> bool:
        return False

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
