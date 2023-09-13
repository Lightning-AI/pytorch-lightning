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
from contextlib import contextmanager
from typing import Any, Generator, Literal, Optional, TYPE_CHECKING
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning.fabric.plugins.precision.precision import Precision


import torch
import torch.utils._device

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", message=r".*bitsandbytes was compiled without GPU support.*")
_BITSANDBYTES_AVAILABLE = RequirementCache("bitsandbytes>=0.40.0")

if TYPE_CHECKING and _BITSANDBYTES_AVAILABLE:
    warnings.filterwarnings(
        "ignore", message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization"
    )
    import bitsandbytes as bnb


log = logging.getLogger(__name__)


class ColBlockQuantizedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias: bool, *, bits, tile_cols):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_cols = tile_cols if tile_cols != -1 else self.in_features
        self.bits = bits
        self.entries_per_byte = 8 // bits
        assert self.entries_per_byte > 0
        assert self.entries_per_byte * self.bits == 8
        assert in_features % self.entries_per_byte == 0
        self.register_buffer(
            "quant_weight",
            torch.empty((self.out_features, self.in_features // self.entries_per_byte), dtype=torch.uint8)
            .t()
            .contiguous()
            .t(),
        )
        self.register_buffer(
            "scales", torch.empty((self.out_features, (self.in_features + self.tile_cols - 1) // self.tile_cols))
        )
        self.register_buffer("zeros", torch.empty_like(self.scales))
        assert isinstance(bias, bool)
        if bias:
            self.register_buffer("bias", torch.empty((self.out_features,)))
        else:
            self.register_buffer("bias", None)

    def pack_weight(self, weight):
        weight = weight.to(device=self.quant_weight.device, copy=True)
        for j in range(self.scales.size(1)):
            weight[:, j * self.tile_cols : (j + 1) * self.tile_cols] /= self.scales[:, j : j + 1]
            weight[:, j * self.tile_cols : (j + 1) * self.tile_cols] += self.zeros[:, j : j + 1]
        weight = weight.clamp_(min=0, max=2**self.bits - 1).to(dtype=torch.uint8)
        self.quant_weight.zero_()
        for nr in range(self.entries_per_byte):
            self.quant_weight += weight[:, nr :: self.entries_per_byte] << (nr * self.bits)

    def get_weight(self, dtype=torch.float):
        weight = torch.empty((self.out_features, self.in_features), device=self.quant_weight.device, dtype=dtype)
        mask = (1 << self.bits) - 1
        for nr in range(self.entries_per_byte):
            weight[:, nr :: self.entries_per_byte] = ((self.quant_weight >> (nr * self.bits)) & mask).float()
        self.quant_weight.to(dtype)
        for j in range(self.scales.size(1)):
            weight[:, j * self.tile_cols : (j + 1) * self.tile_cols] -= self.zeros[:, j : j + 1]
            weight[:, j * self.tile_cols : (j + 1) * self.tile_cols] *= self.scales[:, j : j + 1]
        return weight

    def forward(self, inp):
        if (
            triton is not None
            and self.bits == 4
            and self.quant_weight.device.type == "cuda"
            and self.zeros.shape[1] == 1
            and self.quant_weight.shape[1] % 32 == 0
        ):
            return qlinear_4bit_weight(inp, self.quant_weight, self.scales, self.zeros)
        weight = self.get_weight(dtype=inp.dtype)
        return torch.nn.functional.linear(inp, weight, self.bias)


class InferenceLinear8bitLt(bnb.nn.Linear8bitLt):
    """Wraps `bnb.nn.Linear8bitLt` and enables instantiation directly on the device and
    re-quantizaton when loading the state dict.


    This should only be used for inference. For training, use `bnb.nn.Linear8bitLt` directly.
    """

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


class BitsandbytesPrecision(Precision):
    precision: Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]

    def __init__(self, mode: Literal):
        if not _BITSANDBYTES_AVAILABLE:
            raise ModuleNotFoundError(str(_BITSANDBYTES_AVAILABLE))

        self.mode = mode

    @contextmanager
    def init_context(self):
        if self.mode is None:
            yield
            return

        if self.mode == "bnb.int8":
            quantized_linear_cls = InferenceLinear8bitLt
        elif self.mode == "bnb.fp4":
            # Use a class instead `functools.partial` to respect `isinstance` checks and attribute accesses
            class QuantizedLinear(Linear4bit):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, quant_type="fp4", compress_statistics=False, **kwargs)

            quantized_linear_cls = QuantizedLinear
        elif self.mode == "bnb.fp4-dq":

            class QuantizedLinear(Linear4bit):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, quant_type="fp4", compress_statistics=True, **kwargs)

            quantized_linear_cls = QuantizedLinear
        elif self.mode == "bnb.nf4":

            class QuantizedLinear(Linear4bit):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, quant_type="nf4", compress_statistics=False, **kwargs)

            quantized_linear_cls = QuantizedLinear
        elif self.mode == "bnb.nf4-dq":

            class QuantizedLinear(Linear4bit):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, quant_type="nf4", compress_statistics=True, **kwargs)

            quantized_linear_cls = QuantizedLinear
        elif self.mode == "gptq.int4":

            class QuantizedLinear(ColBlockQuantizedLinear):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, bits=4, tile_cols=-1, **kwargs)

            quantized_linear_cls = QuantizedLinear
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")

        torch_linear_cls = torch.nn.Linear
        torch.nn.Linear = quantized_linear_cls
        yield
        torch.nn.Linear = torch_linear_cls

    def init_model(self):
        ...
