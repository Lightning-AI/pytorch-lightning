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
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Literal

import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from lightning.fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning.pytorch.plugins.precision.precision import Precision


class HalfPrecision(Precision):
    """Plugin for training with half precision.

    Args:
        precision: Whether to use ``torch.float16`` (``'16-true'``) or ``torch.bfloat16`` (``'bf16-true'``).

    """

    precision: Literal["bf16-true", "16-true"] = "16-true"

    def __init__(self, precision: Literal["bf16-true", "16-true"] = "16-true") -> None:
        self.precision = precision
        self._desired_input_dtype = torch.bfloat16 if precision == "bf16-true" else torch.float16

    @override
    def convert_module(self, module: Module) -> Module:
        return module.to(dtype=self._desired_input_dtype)

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def module_init_context(self) -> AbstractContextManager:
        return self.tensor_init_context()

    @override
    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type when tensors get created during the module's forward.

        See: :meth:`torch.set_default_tensor_type`

        """
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self._desired_input_dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(default_dtype)

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)
