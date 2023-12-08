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
from typing import Literal

import torch
from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.plugins.precision.precision import Precision

_ITREX_AVAILABLE = RequirementCache("intel-extension-for-transformers>=1.2.2")


class ITREXPrecision(Precision):
    """Plugin for quantizing weights with `intel-extension-for-transformers <https://github.com/intel/intel-extension-for-transformers>`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        mode: The quantization mode to use.
    """

    def __init__(
        self,
        mode: Literal["int8", "int4_fullrange", "int4_clip", "nf4", "fp4_e2m1"],
    ) -> None:
        super().__init__()
        self.mode = mode

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        if not _ITREX_AVAILABLE:
            raise ModuleNotFoundError(str(_ITREX_AVAILABLE))

        from intel_extension_for_transformers.llm.quantization.utils import convert_to_quantized_model
        from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig

        config = WeightOnlyQuantConfig(weight_dtype=self.mode)
        module.eval()
        config.post_init()
        module = convert_to_quantized_model(module, config)
        return module
