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
from typing import Any, Optional, Sequence

from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _HPU_AVAILABLE

if _HPU_AVAILABLE:
    from habana_frameworks.torch.hpex import hmp


class HPUPrecisionPlugin(PrecisionPlugin):
    """Plugin that enables bfloats/floats on HPUs."""

    def __init__(self, precision: int, hmp_params: Optional[Sequence[Any]] = None) -> None:
        if not _HPU_AVAILABLE:
            raise MisconfigurationException("HPU precision plugin requires HPU devices.")
        super().__init__()
        self.precision = precision
        if not hmp_params:
            return

        hmp_opt_level = hmp_params.get("level", "02")  # type: ignore
        hmp_bf16 = hmp_params.get("bf16_ops", None)  # type: ignore
        hmp_fp32 = hmp_params.get("fp32_ops", None)  # type: ignore
        hmp_verbose = hmp_params.get("verbose", False)  # type: ignore

        hmp.convert(opt_level=hmp_opt_level, bf16_file_path=hmp_bf16, fp32_file_path=hmp_fp32, isVerbose=hmp_verbose)
