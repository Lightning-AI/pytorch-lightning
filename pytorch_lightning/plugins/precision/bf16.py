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
from contextlib import contextmanager
from typing import Generator

import torch

from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin
from pytorch_lightning.utilities import _TORCH_BFLOAT_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TORCH_BFLOAT_AVAILABLE:
    from torch import autocast


class Bf16PrecisionPlugin(MixedPrecisionPlugin):
    precision: str = "bf16"

    def __init__(self, use_cpu: bool) -> None:
        super().__init__()
        if not _TORCH_BFLOAT_AVAILABLE:
            raise MisconfigurationException("To use `precision='bf16' you must install torch greater or equal to 1.10.")
        self.use_cpu = use_cpu

    def autocast_context_manager(self) -> autocast:
        return autocast("cpu" if self.use_cpu else "cuda", dtype=torch.bfloat16)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield
