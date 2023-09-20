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

# Based on original implementation in Lit-GPT by
# Thomas Viehmann, Carlos Mocholí, and Adrian Wälchli

import logging
import os
import warnings
from contextlib import contextmanager
from typing import Literal, TYPE_CHECKING

import torch
import torch.utils._device
from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.plugins.precision.precision import Precision

_TRITON_AVAILABLE = RequirementCache("triton")
if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", message=r".*bitsandbytes was compiled without GPU support.*")
_BITSANDBYTES_AVAILABLE = RequirementCache("bitsandbytes>=0.40.0")

if TYPE_CHECKING and _BITSANDBYTES_AVAILABLE:
    warnings.filterwarnings(
        "ignore", message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization"
    )
    import bitsandbytes as bnb


log = logging.getLogger(__name__)


def qlinear_4bit_weight(inp, weight, scales, zeros):
    weight = weight.t().contiguous()
    c_shape = inp.shape[:-1] + weight.shape[-1:]
    inp = inp.reshape(-1, inp.shape[-1]).contiguous()
    # we pad the input to amortize triton compilation cost better
    PAD_TO = 256
    if inp.shape[0] % PAD_TO != 0:
        c_crop = inp.shape[0]
        new_inp_shape0 = inp.shape[0] + PAD_TO - inp.shape[0] % PAD_TO
        inp2 = inp.new_empty((new_inp_shape0, inp.shape[1]))
        inp2[: inp.shape[0]] = inp
        inp2[inp.shape[0] :].zero_()
        inp = inp2
    else:
        c_crop = None

    assert inp.shape[1] == weight.shape[0] * 2, "incompatible dimensions"

    assert scales.shape == (weight.shape[1], 1)
    assert zeros.shape == (weight.shape[1], 1)
    scales = scales.contiguous()
    zeros = zeros.contiguous()
    K, N = weight.shape
    M, K = inp.shape
    assert K % 32 == 0, "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    linear_kernel_4bit_weight[grid](
        inp,
        weight,
        c,
        scales,
        zeros,
        M,
        N,
        K,
        inp.stride(0),
        inp.stride(1),
        weight.stride(0),
        weight.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c[:c_crop].reshape(c_shape)


@triton.jit
def linear_kernel_4bit_weight(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    bscales_ptr,
    bzeros_ptr,
    # bdequant,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.T.

    A has shape (M, K), B has shape (N, K) and C has shape (M, N)

    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
    # see above `Pointer Arithmetics` section for details
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * stride_bn)

    bscales_ptrs = bscales_ptr + offs_bn[None, :]
    bzeros_ptrs = bzeros_ptr + offs_bn[None, :]

    scale = tl.load(bscales_ptrs)
    zero = tl.load(bzeros_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # wasteful as it is to load everything twice, my attempts at avoiding it lead to slower code
        b12 = tl.load(b_ptrs, mask=b_mask)
        # Note that for simplicity, we don't apply a mask in K here.
        a = tl.load(a_ptrs, mask=a_mask).to(tl.float32)
        b = (((b12.to(tl.uint8) >> ((offs_k[:, None] % 2) * 4)) & 0xF).to(tl.float32) - zero) * scale
        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


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


class BitsandbytesQuantization(Precision):
    """Plugin for training with bitsandbytes quantized weights.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        mode: the bitsandbytes quantization dtype to use.

    .. note:: bitsandbytes only supports Linux environments.

    """

    precision: Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]

    def __init__(self, mode: Literal):
        if not _BITSANDBYTES_AVAILABLE:
            raise ModuleNotFoundError(str(_BITSANDBYTES_AVAILABLE))

        self.mode = mode

    def convert_module(self) -> None:
        # TODO check if bitsandbytes already handles conversion
        ...

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
            raise ValueError(f"Unknown quantization mode: {self.mode}")

        torch_linear_cls = torch.nn.Linear
        torch.nn.Linear = quantized_linear_cls
        yield
        torch.nn.Linear = torch_linear_cls


def _convert_layers(module: torch.nn.Module) -> None:
    # TODO check if bitsandbytes already handles conversion
    ...
