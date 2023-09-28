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
from contextlib import ExitStack
from types import ModuleType
from typing import Any, ContextManager, Dict, Literal, Optional, Set, Type

import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.utils import (
    _ClassReplacementContextManager,
    _convert_fp_tensor,
    _DtypeContextManager,
)

log = logging.getLogger(__name__)

_BITSANDBYTES_AVAILABLE = RequirementCache("bitsandbytes>=0.41.0")


class BitsandbytesPrecision(Precision):
    """Plugin for quantizing weights with
    `bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    .. note::
        The optimizer is not automatically replaced with ``bitsandbytes.optim.Adam8bit`` or equivalent 8-bit optimizers.

    Args:
        mode: The quantization mode to use.
        dtype: The compute dtype to use.
        skips: The submodules whose Linear layers should not be replaced, for example. ``{"transformer.lm_head"}``.
            This might be desirable for numerical stability. The string will be checked in as a prefix, so a value like
            "transformer.blocks" will skip all linear layers in all of the transformer blocks.
    """

    # Note: you'll notice that the `precision` str class attribute is not defined. This is on purpose because there are
    # many configuration options so `precision="bitsandbytes"` would be ambiguous about which one to use. Additionally,
    # it would create backwards compatibility challenges if better modes or dtypes are added in the future

    # TODO: we could implement optimizer replacement with
    # - Fabric: Add `Precision.convert_optimizer` from `Strategy.setup_optimizer`
    # - Trainer: Use `PrecisionPlugin.connect`

    def __init__(
        self,
        mode: Literal["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"],
        dtype: Optional[torch.dtype] = None,
        skips: Optional[Set[str]] = None,
    ) -> None:
        if not _BITSANDBYTES_AVAILABLE:
            raise ModuleNotFoundError(str(_BITSANDBYTES_AVAILABLE))

        if dtype is None:
            # try to be smart about the default selection
            if mode.startswith("int8"):
                dtype = torch.float16
            else:
                dtype = (
                    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                )
        if mode.startswith("int8") and dtype is not torch.float16:
            # this limitation is mentioned in https://huggingface.co/blog/hf-bitsandbytes-integration#usage
            raise ValueError(f"{mode!r} only works with `dtype=torch.bfloat16`, but you chose `{dtype}`")

        mode_to_cls = {
            "nf4": _NF4Linear,
            "nf4-dq": _NF4DQLinear,
            "fp4": _FP4Linear,
            "fp4-dq": _FP4DQLinear,
            "int8-training": _Linear8bitLt,
            "int8": _Int8LinearInference,
        }
        self._linear_cls = mode_to_cls[mode]
        self.dtype = dtype
        self.skips = skips or set()

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # avoid naive users thinking they quantized their model
        if not any(isinstance(m, torch.nn.Linear) for m in module.modules()):
            raise TypeError(
                "You are using the bitsandbytes precision plugin, but your model has no Linear layers. This plugin"
                " won't work for your model."
            )

        # convert modules if they haven't been converted already
        if not any("bitsandbytes.nn" in m.__module__ for m in module.modules()):
            _convert_layers(module, self._linear_cls, self.skips)

        # set the compute dtype if necessary
        for m in module.modules():
            if isinstance(m, _Linear4bit):
                m.compute_dtype = self.dtype
                m.compute_type_is_set = False
        return module

    def init_context(self) -> ContextManager:
        dtype_ctx = _DtypeContextManager(self.dtype)
        if self.skips:
            # cannot patch the Linear class if the user wants to skip some submodules
            return dtype_ctx
        stack = ExitStack()
        stack.enter_context(dtype_ctx)
        # TODO: this could also consider replacing with `bnb.nn.Embedding`
        context_manager = _ClassReplacementContextManager({"torch.nn.Linear": self._linear_cls})
        stack.enter_context(context_manager)
        return stack

    def forward_context(self) -> ContextManager:
        return _DtypeContextManager(self.dtype)

    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self.dtype)

    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())


def _import_bitsandbytes() -> ModuleType:
    if not _BITSANDBYTES_AVAILABLE:
        raise ModuleNotFoundError(str(_BITSANDBYTES_AVAILABLE))
    # configuration for bitsandbytes before import
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    warnings.filterwarnings("ignore", message=r".*bitsandbytes was compiled without GPU support.*")
    warnings.filterwarnings(
        "ignore", message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization"
    )
    import bitsandbytes

    return bitsandbytes


if _BITSANDBYTES_AVAILABLE:
    bnb = _import_bitsandbytes()

    class _Linear8bitLt(bnb.nn.Linear8bitLt):  # type: ignore[name-defined]
        """Wraps `bnb.nn.Linear8bitLt` and enables instantiation directly on the device and
        re-quantizaton when loading the state dict.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("threshold", 6.0)
            super().__init__(*args, **kwargs)
            # We quantize the initial weight here so we don't end up filling the device
            # memory with float32 weights which could lead to OOM.
            self._quantize_weight(self.weight.data)

        def _load_from_state_dict(self, local_state_dict: Dict, *args: Any, **kwargs: Any) -> None:
            # There is only one key that ends with `*.weight`, the other one is the bias
            weight_key = next((name for name in local_state_dict if name.endswith("weight")), None)
            if weight_key is None:
                return

            # Load the weight from the state dict and re-quantize it
            weight = local_state_dict.pop(weight_key)
            self._quantize_weight(weight)

            # If there is a bias, let nn.Module load it
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

    class _Linear4bit(bnb.nn.Linear4bit):  # type: ignore[name-defined]
        def __init__(self, *args: Any, device: Optional[torch.device] = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if device is None:
                device = torch.tensor(0).device
            if device.type == "cuda":
                # weight needs to be moved manually because it doesn't work with device as a context manager:
                # `weight.to()` gets skipped if `weight.data` is already on CUDA. we avoid it by moving it back
                # (inefficient). see condition:
                # https://github.com/TimDettmers/bitsandbytes/blob/817bdf6/bitsandbytes/nn/modules.py#L177
                self.weight.data = self.weight.data.to("cpu")
                warnings.filterwarnings("ignore", message=r".*Fabric.setup\(\)` has parameters on different devices.*")
                # we could manually move `self.weight.to(device)` here but that breaks checkpoint loading
                # bnb expects that the layers are moved to the device after loading

    # Use a class instead `functools.partial` to respect `isinstance` checks and attribute accesses
    class _Int8LinearInference(_Linear8bitLt):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("has_fp_weights", False)
            super().__init__(*args, **kwargs)

    class _FP4Linear(_Linear4bit):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("quant_type", "fp4")
            kwargs.setdefault("compress_statistics", False)
            super().__init__(*args, **kwargs)

    class _FP4DQLinear(_Linear4bit):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("quant_type", "fp4")
            kwargs.setdefault("compress_statistics", True)
            super().__init__(*args, **kwargs)

    class _NF4Linear(_Linear4bit):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("quant_type", "nf4")
            kwargs.setdefault("compress_statistics", False)
            super().__init__(*args, **kwargs)

    class _NF4DQLinear(_Linear4bit):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("quant_type", "nf4")
            kwargs.setdefault("compress_statistics", True)
            super().__init__(*args, **kwargs)


def _convert_layers(module: torch.nn.Module, linear_cls: Type, skips: Set[str]) -> None:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and not any(name.startswith(s) for s in skips):
            has_bias = child.bias is not None
            replacement = linear_cls(child.in_features, child.out_features, bias=has_bias)
            replacement.weight.data = child.weight.data.clone()
            if has_bias:
                replacement.bias.data = child.bias.data.clone()
            log.debug(f"Replacing layer {name!r} with bitsandbytes equivalent")
            module.__setattr__(name, replacement)
        else:
            _convert_layers(child, linear_cls, skips)