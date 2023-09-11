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
import logging
import os
import warnings
from typing import Literal

import torch
from lightning_utilities.core.imports import RequirementCache

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", message=r".*bitsandbytes was compiled without GPU support.*")
_BITSANDBYTES_AVAILABLE = RequirementCache("bitsandbytes>=0.40.0")

if _BITSANDBYTES_AVAILABLE:
    warnings.filterwarnings(
        "ignore", message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization"
    )
    import bitsandbytes as bnb


log = logging.getLogger(__name__)


class InferenceLinear8bitLt(bnb.nn.Linear8bitLt):
    """Wraps `bnb.nn.Linear8bitLt` and enables instantiation directly on the device and
    re-quantizaton when loading the state dict.


    This should only be used for inference. For training, use `bnb.nn.Linear8bitLt` directly.
    """

    quantize: Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            has_fp16_weights=False,
            threshold=6.0,
        )

        self._quantize_weight(self.weight.data)

    def _load_from_state_dict(self, local_state_dict, *args, **kwargs):
        weight_key = next((name for name in local_state_dict if name.endswith("weight")), None)
        if weight_key is None:
            return

        weight = local_state_dict.pop(weight_key)
        self._quantize_weight(weight)

        if local_state_dict:
            super()._load_from_state_dict(local_state_dict, *args, **kwargs)

    def _quantize_weight(self, weight: torch.Tensor) -> None:
        # This code is taken and adapted from `bnb.nn.Int8Params.cuda()`
        B = weight.contiguous().half().cuda()
        CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
        del CBt
        del SCBt
        self.weight.data = CB
        setattr(self.weight, "CB", CB)
        setattr(self.weight, "SCB", SCB)


class Linear4bit(bnb.modules.Linear4bit):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        if device is None:
            device = torch.tensor(0).device
        if device.type == "cuda":
            self.weight.data = self.weight.data.to("cpu")
            warnings.filterwarnings("ignore", message=r".*Fabric.setup\(\)` has parameters on different devices.*")
