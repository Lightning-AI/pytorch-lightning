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
import os
from typing import Any, List, Tuple

import torch.nn as nn
from habana_frameworks.torch.hpex import hmp
from torch.optim import Optimizer

from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin


class HPUPrecisionPlugin(PrecisionPlugin):
    """Plugin that enables bfloats/floats on HPUs."""

    def __init__(self, precision: int, hmp_params: []) -> None:
        super().__init__()
        self.precision = precision
        if hmp_params is not None:
            hmp_opt_level = hmp_params["level"]
            hmp_bf16 = hmp_params["bf16_ops"]
            hmp_fp32 = hmp_params["fp32_ops"]
            hmp_verbose = hmp_params["verbose"]
            hmp.convert(
                opt_level=hmp_opt_level, bf16_file_path=hmp_bf16, fp32_file_path=hmp_fp32, isVerbose=hmp_verbose
            )

    def connect(
        self, model: nn.Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
    ) -> Tuple[nn.Module, List[Optimizer], List[Any]]:
        return super().connect(model=model, optimizers=optimizers, lr_schedulers=lr_schedulers)
