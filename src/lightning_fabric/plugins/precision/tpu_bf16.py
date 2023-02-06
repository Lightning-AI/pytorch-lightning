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
import os

import torch
from torch import Tensor
from typing_extensions import Literal

from lightning_fabric.plugins.precision import TPUPrecision
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor


class TPUBf16Precision(TPUPrecision):
    """Plugin that enables bfloats on TPUs."""

    precision: Literal["bf16"] = "bf16"

    def __init__(self) -> None:
        super().__init__()
        os.environ["XLA_USE_BF16"] = "1"

    def convert_input(self, data: Tensor) -> Tensor:
        return _convert_fp_tensor(data, torch.bfloat16)

    def teardown(self) -> None:
        os.environ.pop("XLA_USE_BF16", None)
