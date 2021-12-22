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
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.plugins.io.xla_plugin import XLACheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.training_type.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities import _TPU_AVAILABLE, find_shared_parameters, set_shared_parameters
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


class SingleTPUStrategy(SingleDeviceStrategy):
    """Strategy for training on a single TPU device."""

    def __init__(
        self,
        device: int,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        checkpoint_io: Optional[XLACheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
    ):
        device = xm.xla_device(device)
        checkpoint_io = checkpoint_io or XLACheckpointIO()
        super().__init__(
            accelerator=accelerator, device=device, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin
        )

        self.debug = debug
        self.tpu_local_core_rank = 0
        self.tpu_global_core_rank = 0

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer: "pl.Trainer") -> None:
        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        if is_overridden("on_post_move_to_device", self.lightning_module):
            self.model.on_post_move_to_device()
        else:
            set_shared_parameters(self.model, shared_params)

        super().setup(trainer)

        if isinstance(self.device, int):
            self.device = xm.xla_device(self.device)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = str(1)

        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()

    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    def teardown(self) -> None:
        # TPU teardown
        os.environ.pop("PT_XLA_DEBUG", None)

    @SingleDeviceStrategy.checkpoint_io.setter
    def checkpoint_io(self, io: Optional[XLACheckpointIO]) -> None:
        if io is not None and not isinstance(io, XLACheckpointIO):
            raise MisconfigurationException(f"{self.__class__.__name__}.checkpoint_io` must be a `XLACheckpointIO`.")
        self._checkpoint_io = io
