# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#

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


class HPUPrecisionPlugin(PrecisionPlugin):
    """Plugin that enables bfloats/floats on HPUs."""

    def __init__(self, precision: int, hmp_params: Optional[Sequence[Any]] = None) -> None:
        if not _HPU_AVAILABLE:
            raise MisconfigurationException("HPU Accelerator requires HPU devices to run ."
                                            "HPU precision plugin requires HPU support ")
        super().__init__()
        self.precision = precision
        if hmp_params is not None:

            from habana_frameworks.torch.hpex import hmp

            hmp_opt_level = hmp_params["level"]  # type: ignore
            hmp_bf16 = hmp_params["bf16_ops"]  # type: ignore
            hmp_fp32 = hmp_params["fp32_ops"]  # type: ignore
            hmp_verbose = hmp_params["verbose"]  # type: ignore
            hmp.convert(
                opt_level=hmp_opt_level, bf16_file_path=hmp_bf16, fp32_file_path=hmp_fp32, isVerbose=hmp_verbose
            )
