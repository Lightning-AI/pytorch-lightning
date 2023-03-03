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
from typing import cast, Literal, Optional

from typing_extensions import get_args

from lightning.pytorch.accelerators.hpu import _HPU_AVAILABLE
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
from lightning.pytorch.utilities.exceptions import MisconfigurationException

if _HPU_AVAILABLE:
    from habana_frameworks.torch.hpex import hmp

_PRECISION_INPUT = Literal["32-true", "16-mixed", "bf16-mixed"]


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
        precision: _PRECISION_INPUT,
        opt_level: str = "O2",
        bf16_file_path: Optional[str] = None,
        fp32_file_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if not _HPU_AVAILABLE:
            raise MisconfigurationException("HPU precision plugin requires HPU devices.")
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(accelerator='hpu', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision}."
            )
        self.precision = cast(_PRECISION_INPUT, str(precision))
        if self.precision in ("16-mixed", "bf16-mixed"):
            hmp.convert(
                opt_level=opt_level, bf16_file_path=bf16_file_path, fp32_file_path=fp32_file_path, isVerbose=verbose
            )
