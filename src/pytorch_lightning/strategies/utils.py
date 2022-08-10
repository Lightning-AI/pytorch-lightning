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

import torch

from pytorch_lightning.utilities.enums import PrecisionType


def on_colab_kaggle() -> bool:
    return bool(os.getenv("COLAB_GPU") or os.getenv("KAGGLE_URL_BASE"))


def _fp_to_half(tensor: torch.Tensor, precision: PrecisionType) -> torch.Tensor:
    if torch.is_floating_point(tensor):
        if precision == PrecisionType.HALF:
            return tensor.half()
        if precision == PrecisionType.BFLOAT:
            return tensor.bfloat16()

    return tensor
