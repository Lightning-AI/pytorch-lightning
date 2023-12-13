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
import functools
import logging
import math
import os
import warnings
from contextlib import ExitStack
from functools import partial
from types import ModuleType
from typing import Any, Callable, ContextManager, Literal, Optional, OrderedDict, Set, Tuple, Type, cast

import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import init
from torch.nn.modules.module import _IncompatibleKeys
from typing_extensions import Self

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.utils import (
    _ClassReplacementContextManager,
    _convert_fp_tensor,
    _DtypeContextManager,
)
from lightning.fabric.utilities.types import _DEVICE

log = logging.getLogger(__name__)

# TODO: unpin after resolving the `quant_state` format breaking changes
_BITSANDBYTES_AVAILABLE = RequirementCache("bitsandbytes==0.41.0")


class BitsandbytesPrecision(Precision):
    """Plugin for quantizing weights with `bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    .. note::
        The optimizer is not automatically replaced with ``bitsandbytes.optim.Adam8bit`` or equivalent 8-bit optimizers.

    Args:
        mode: The quantization mode to use.
        dtype: The compute dtype to use.
        ignore_modules: The submodules whose Linear layers should not be replaced, for example. ``{"lm_head"}``.
            This might be desirable for numerical stability. The string will be checked in as a prefix, so a value like
            "transformer.blocks" will ignore all linear layers in all of the transformer blocks.
    """

    # Note: you'll notice that the `precision` str class attribute is not defined. This is on purpose because there are
    # many configuration options so `precision="bitsandbytes"` would be ambiguous about which one to use. Additionally,
    # it would create backwards compatibility challenges if better modes or dtypes are added in the future

    # TODO: we could implement optimizer replacement with
    # - Fabric: Add `Precision.convert_optimizer` from `Strategy.setup_optimizer`
    # - Trainer: Use `Precision.connect`

    def __init__(
        self,
        mode: Literal["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"],
        dtype: Optional[torch.dtype] = None,
        ignore_modules: Optional[Set[str]] = None,
    ) -> None:
        _import_bitsandbytes()

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
            raise ValueError(f"{mode!r} only works with `dtype=torch.float16`, but you chose `{dtype}`")

        globals_ = globals()
        mode_to_cls = {
            "nf4": globals_["_NF4Linear"],
            "nf4-dq": globals_["_NF4DQLinear"],
            "fp4": globals_["_FP4Linear"],
            "fp4-dq": globals_["_FP4DQLinear"],
            "int8-training": globals_["_Linear8bitLt"],
            "int8": globals_["_Int8LinearInference"],
        }
        self._linear_cls = mode_to_cls[mode]
        self.dtype = dtype
        self.ignore_modules = ignore_modules or set()

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # avoid naive users thinking they quantized their model
        if not any(isinstance(m, torch.nn.Linear) for m in module.modules()):
            raise TypeError(
                "You are using the bitsandbytes precision plugin, but your model has no Linear layers. This plugin"
                " won't work for your model."
            )

        # convert modules if they haven't been converted already
        bnb = _import_bitsandbytes()
        if not any(isinstance(m, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)) for m in module.modules()):
            _convert_layers(module, self._linear_cls, self.ignore_modules)

        # set the compute dtype if necessary
        for m in module.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                m.compute_dtype = self.dtype
                m.compute_type_is_set = False
        return module

    def tensor_init_context(self) -> ContextManager:
        return _DtypeContextManager(self.dtype)

    def module_init_context(self) -> ContextManager:
        if self.ignore_modules:
            # cannot patch the Linear class if the user wants to skip some submodules
            raise RuntimeError(
                "Instantiating your model under the `init_module` context manager is not supported when used with"
                f" `BitsandbytesPrecision(..., ignore_modules={self.ignore_modules})` as this"
                " may initialize the layers on-device, defeating the purpose of quantization. You can remove"
                " `ignore_modules` or remove the `init_module` context manager."
            )
        dtype_ctx = self.tensor_init_context()
        # TODO: this could also support replacing `Embedding` and `Conv1D`
        context_manager = _ClassReplacementContextManager({"torch.nn.Linear": self._linear_cls})
        stack = ExitStack()
        stack.enter_context(dtype_ctx)
        stack.enter_context(context_manager)
        return stack

    def forward_context(self) -> ContextManager:
        return _DtypeContextManager(self.dtype)

    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self.dtype)

    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())


def _quantize_on_load_hook(quantize_fn: Callable[[torch.Tensor], None], state_dict: OrderedDict, *_: Any) -> None:
    # There is only one key that ends with `*.weight`, the other one is the bias
    weight_key = next((name for name in state_dict if name.endswith("weight")), None)
    if weight_key is None:
        return
    # Load the weight from the state dict and re-quantize it
    weight = state_dict.pop(weight_key)
    quantize_fn(weight)


def _ignore_missing_weights_hook(module: torch.nn.Module, incompatible_keys: _IncompatibleKeys) -> None:
    for key in reversed(incompatible_keys.missing_keys):
        if key.endswith("weight"):
            incompatible_keys.missing_keys.remove(key)


@functools.lru_cache(maxsize=1)
def _import_bitsandbytes() -> ModuleType:
    if not _BITSANDBYTES_AVAILABLE:
        raise ModuleNotFoundError(str(_BITSANDBYTES_AVAILABLE))
    # configuration for bitsandbytes before import
    nowelcome_set = "BITSANDBYTES_NOWELCOME" in os.environ
    if not nowelcome_set:
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    warnings.filterwarnings("ignore", message=r".*bitsandbytes was compiled without GPU support.*")
    warnings.filterwarnings(
        "ignore", message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization"
    )
    import bitsandbytes as bnb

    if not nowelcome_set:
        del os.environ["BITSANDBYTES_NOWELCOME"]

    def _replace_param(
        p: torch.nn.Parameter, data: torch.Tensor, quant_state: Optional[Tuple] = None
    ) -> torch.nn.Parameter:
        # doing `param.data = weight` raises a RuntimeError if param.data was on meta-device, so
        # we need to re-create the parameters instead of overwriting the data
        if p.device.type == "meta":
            if isinstance(p, bnb.nn.Params4bit):
                return bnb.nn.Params4bit(
                    data,
                    requires_grad=False,
                    quant_state=quant_state,
                    compress_statistics=p.compress_statistics,
                    quant_type=p.quant_type,
                )
            return torch.nn.Parameter(data, requires_grad=p.requires_grad)
        p.data = data
        if isinstance(p, bnb.nn.Params4bit):
            p.quant_state = quant_state
        return p

    class _Linear8bitLt(bnb.nn.Linear8bitLt):
        """Wraps `bnb.nn.Linear8bitLt` and enables instantiation directly on the device and re-quantizaton when loading
        the state dict."""

        def __init__(self, *args: Any, device: Optional[_DEVICE] = None, threshold: float = 6.0, **kwargs: Any) -> None:
            super().__init__(*args, device=device, threshold=threshold, **kwargs)
            self.weight = cast(bnb.nn.Int8Params, self.weight)
            # if the device is CUDA or we are under a CUDA context manager, quantize the weight here, so we don't end up
            # filling the device memory with float32 weights which could lead to OOM
            if torch.tensor(0, device=device).device.type == "cuda":
                self.quantize()
            self._register_load_state_dict_pre_hook(partial(_quantize_on_load_hook, self.quantize))
            self.register_load_state_dict_post_hook(_ignore_missing_weights_hook)

        def quantize(self, weight: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> None:
            if weight is None:
                if weight.data.type == torch.int8:
                    # already quantized
                    return
                weight = self.weight.data
            assert isinstance(self.weight, bnb.nn.Int8Params)
            self.weight = self._quantize(self.weight, weight, device)

        @staticmethod
        def _quantize(
            int8params: bnb.nn.Int8Params, weight: torch.Tensor, device: Optional[torch.device]
        ) -> bnb.nn.Int8Params:
            if device.type == "meta":
                # need custom logic if int8params is on meta device
                raise NotImplementedError
            # https://github.com/TimDettmers/bitsandbytes/blob/0.41.0/bitsandbytes/nn/modules.py#L291-L302
            B = weight.contiguous().to(device="cuda", dtype=torch.float16)
            if int8params.has_fp16_weights:
                int8params.data = B
            else:
                CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
                del CBt
                del SCBt
                int8params.data = CB
                setattr(int8params, "CB", CB)
                setattr(int8params, "SCB", SCB)
            return int8params

        def to_empty(self, *, device: _DEVICE, recurse: bool = True) -> Self:
            if self.weight.device.type == "meta":
                # need custom logic if int8params is on meta device
                raise NotImplementedError
            if self.weight.dtype == torch.uint8:  # was quantized
                # need the original shape here
                raise NotImplementedError
            weight = torch.empty_like(self.weight.data, device=device)
            if torch.device(device).type == "cuda":  # re-quantize
                self.quantize(weight, device)
            else:
                self.weight = _replace_param(self.weight, weight)
            if self.bias is not None:
                self.bias = _replace_param(self.bias, torch.empty_like(self.bias, device=device))
            return self

        def reset_parameters(self):
            # from `torch.nn.Linear.reset_parameters`
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)

            linear_init_finished = isinstance(self.weight, bnb.nn.Params4bit)
            if linear_init_finished and self.weight.dtype == torch.uint8:  # was quantized
                # need the original shape here
                raise NotImplementedError
            weight = self.weight.data
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if linear_init_finished:
                if self.weight.device.type == "meta":
                    # need custom logic if int8params is on meta device
                    raise NotImplementedError
                if self.weight.device.type == "cuda":  # re-quantize
                    self.quantize(weight)
                else:
                    self.weight = _replace_param(self.weight, weight)

    class _Linear4bit(bnb.nn.Linear4bit):
        """Wraps `bnb.nn.Linear4bit` to enable: instantiation directly on the device, re-quantizaton when loading the
        state dict, meta-device initialization, and materialization."""

        def __init__(self, *args: Any, device: Optional[_DEVICE] = None, **kwargs: Any) -> None:
            super().__init__(*args, device=device, **kwargs)
            self.weight = cast(bnb.nn.Params4bit, self.weight)
            # if the device is CUDA or we are under a CUDA context manager, quantize the weight here, so we don't end up
            # filling the device memory with float32 weights which could lead to OOM
            if torch.tensor(0, device=device).device.type == "cuda":
                self.quantize()
            self._register_load_state_dict_pre_hook(partial(_quantize_on_load_hook, self.quantize))
            self.register_load_state_dict_post_hook(_ignore_missing_weights_hook)

        def quantize(self, weight: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> None:
            if weight is None:
                if weight.data.type == torch.uint8:
                    # already quantized
                    return
                weight = self.weight.data
            assert isinstance(self.weight, bnb.nn.Params4bit)
            self.weight = self._quantize(self.weight, weight, device)

        @staticmethod
        def _quantize(
            params4bit: bnb.nn.Params4bit, weight: torch.Tensor, device: Optional[torch.device]
        ) -> bnb.nn.Params4bit:
            # https://github.com/TimDettmers/bitsandbytes/blob/0.41.0/bitsandbytes/nn/modules.py#L156-L159
            w = weight.contiguous().to(device=device, dtype=torch.half)
            w_4bit, quant_state = bnb.functional.quantize_4bit(
                w,
                blocksize=params4bit.blocksize,
                compress_statistics=params4bit.compress_statistics,
                quant_type=params4bit.quant_type,
            )
            return _replace_param(params4bit, w_4bit, quant_state)

        def to_empty(self, *, device: _DEVICE, recurse: bool = True) -> Self:
            if self.weight.dtype == torch.uint8:  # was quantized
                # cannot init the quantized params directly
                weight = torch.empty(self.weight.quant_state[1], device=device, dtype=torch.half)
            else:
                weight = torch.empty_like(self.weight.data, device=device)
            if torch.device(device).type == "cuda":  # re-quantize
                self.quantize(weight, device)
            else:
                self.weight = _replace_param(self.weight, weight)
            if self.bias is not None:
                self.bias = _replace_param(self.bias, torch.empty_like(self.bias, device=device))
            return self

        def reset_parameters(self):
            # from `torch.nn.Linear.reset_parameters`
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)

            linear_init_finished = isinstance(self.weight, bnb.nn.Params4bit)
            if linear_init_finished and self.weight.dtype == torch.uint8:  # was quantized
                # cannot init the quantized params directly
                weight = torch.empty(self.weight.quant_state[1], device=self.weight.device, dtype=torch.half)
            else:
                weight = self.weight.data
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if linear_init_finished:
                if self.weight.device.type == "cuda":  # re-quantize
                    self.quantize(weight)
                else:
                    self.weight = _replace_param(self.weight, weight)

    # Use a class instead `functools.partial` to respect `isinstance` checks and attribute accesses
    class _Int8LinearInference(_Linear8bitLt):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, has_fp16_weights=False, **kwargs)

    class _FP4Linear(_Linear4bit):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, quant_type="fp4", compress_statistics=False, **kwargs)

    class _FP4DQLinear(_Linear4bit):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, quant_type="fp4", compress_statistics=True, **kwargs)

    class _NF4Linear(_Linear4bit):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, quant_type="nf4", compress_statistics=False, **kwargs)

    class _NF4DQLinear(_Linear4bit):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, quant_type="nf4", compress_statistics=True, **kwargs)

    # these classes are defined programatically like this to avoid importing bitsandbytes in environments that have
    # it available but will not use it
    classes = {
        "_Linear8bitLt": _Linear8bitLt,
        "_Linear4bit": _Linear4bit,
        "_Int8LinearInference": _Int8LinearInference,
        "_FP4Linear": _FP4Linear,
        "_FP4DQLinear": _FP4DQLinear,
        "_NF4Linear": _NF4Linear,
        "_NF4DQLinear": _NF4DQLinear,
    }
    globals().update(classes)

    return bnb


def _convert_layers(module: torch.nn.Module, linear_cls: Type, ignore_modules: Set[str], prefix: str = "") -> None:
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(fullname.startswith(s) for s in ignore_modules):
            log.debug(f"Replacing layer {fullname!r} with bitsandbytes equivalent")
            has_bias = child.bias is not None
            replacement = linear_cls(
                child.in_features,
                child.out_features,
                bias=has_bias,
                device=torch.device("meta"),
            )
            if has_bias:
                replacement.bias = torch.nn.Parameter(child.bias.data.clone(), requires_grad=child.bias.requires_grad)
            replacement.quantize(child.weight.data.clone())
            module.__setattr__(name, replacement)
        else:
            _convert_layers(child, linear_cls, ignore_modules, prefix=fullname)
