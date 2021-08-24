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

import torch

from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.utilities import _TORCH_CPU_AMP_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CPUNativeMixedPrecisionPlugin(NativeMixedPrecisionPlugin):
    """
    Plugin for CPU native mixed precision training with :mod:`torch.cpu.amp`.

    Args:
        precision: Currently only bf16 (torch.bfloat16) is supported.
    """

    def __init__(self, precision: str = "bf16") -> None:
        if not _TORCH_CPU_AMP_AVAILABLE:
            raise MisconfigurationException(
                "You have asked for native AMP on CPU, but AMP is only available on GPU for PyTorch 1.9 "
                "and lower. To use native AMP on CPU, install PyTorch 1.10 or later."
            )
        super().__init__(precision)

    def _select_precision_dtype(self, precision: str = "bf16") -> torch.dtype:
        if not precision == "bf16":
            raise MisconfigurationException(
                "CPU native amp only supports bfloat16. Please pass precision=bf16 to the Trainer."
            )
        return torch.bfloat16

    def autocast_context_manager(self) -> torch.cuda.amp.autocast:
        return torch.cpu.amp.autocast(fast_dtype=self._fast_dtype)
