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
from typing import Optional, Union

from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _HPU_AVAILABLE

if _HPU_AVAILABLE:
    from habana_frameworks.torch.hpex import hmp


class HPUPrecisionPlugin(PrecisionPlugin):
    """Plugin that enables bfloat/half support on HPUs.

    Args:
        precision: The precision to use.
        opt_level: Choose optimization level for hmp.
        bf16_file_path: Path to bf16 ops list in hmp O1 mode.
        fp32_file_path: Path to fp32 ops list in hmp O1 mode.
        verbose: Enable verbose mode for hmp.
    """

    def __init__(
        self,
        precision: Union[str, int],
        opt_level: str = "O2",
        bf16_file_path: Optional[str] = None,
        fp32_file_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if not _HPU_AVAILABLE:
            raise MisconfigurationException("HPU precision plugin requires HPU devices.")
        supported_precision_values = (16, 32, "bf16")
        if precision not in supported_precision_values:
            raise ValueError(
                f"`Trainer(accelerator='hpu', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision_values}."
            )
        super().__init__()
        self.precision = precision
        if precision in (PrecisionType.HALF, PrecisionType.BFLOAT):
            hmp.convert(
                opt_level=opt_level, bf16_file_path=bf16_file_path, fp32_file_path=fp32_file_path, isVerbose=verbose
            )
