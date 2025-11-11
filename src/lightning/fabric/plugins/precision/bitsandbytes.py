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
from contextlib import AbstractContextManager
from typing import Any, Literal, Optional

import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.utilities.imports import _raise_enterprise_not_available

_BITSANDBYTES_AVAILABLE = RequirementCache("bitsandbytes")


class BitsandbytesPrecision(Precision):
    """Plugin for quantizing weights with `bitsandbytes <https://github.com/bitsandbytes-foundation/bitsandbytes>`__.

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
        ignore_modules: Optional[set[str]] = None,
    ) -> None:
        super().__init__()
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.precision.bitsandbytes import (
            BitsandbytesPrecision as EnterpriseBitsandbytesPrecision,
        )

        self.bitsandbytes_impl = EnterpriseBitsandbytesPrecision(mode=mode, dtype=dtype, ignore_modules=ignore_modules)

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return self.bitsandbytes_impl.convert_module(module)

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return self.bitsandbytes_impl.tensor_init_context()

    @override
    def module_init_context(self) -> AbstractContextManager:
        return self.bitsandbytes_impl.module_init_context()

    @override
    def forward_context(self) -> AbstractContextManager:
        return self.bitsandbytes_impl.forward_context()

    @override
    def convert_input(self, data: Any) -> Any:
        return self.bitsandbytes_impl.convert_input(data)

    @override
    def convert_output(self, data: Any) -> Any:
        return self.bitsandbytes_impl.convert_output(data)
