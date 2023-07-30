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
from contextlib import contextmanager
from typing import Any, cast, Generator, List, Literal, Tuple

import torch
from torch.nn import Module
from torch.optim import Optimizer

import lightning.pytorch as pl
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin


class HalfPrecisionPlugin(PrecisionPlugin):
    """Plugin for training with half precision.

    Args:
        precision: Whether to use ``torch.float16`` (``'16-true'``) or ``torch.bfloat16`` (``'bf16-true'``).
    """

    precision: Literal["bf16-true", "16-true"] = "16-true"

    def __init__(self, precision: Literal["bf16-true", "16-true"] = "16-true") -> None:
        self.precision = precision
        self._desired_input_dtype = torch.bfloat16 if precision == "bf16-true" else torch.float16

    def convert_module(self, module: Module) -> Module:
        return module.to(dtype=self._desired_input_dtype)

    def connect(
        self, model: Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
    ) -> Tuple[Module, List["Optimizer"], List[Any]]:
        """Converts the model parameters to half precision.

        Does not alter `optimizers` or `lr_schedulers`.
        """
        model = cast(pl.LightningModule, self.convert_module(model))
        return super().connect(model, optimizers, lr_schedulers)

    @contextmanager
    def init_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type when initializing module parameters or tensors.

        See: :meth:`torch.set_default_dtype`
        """
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self._desired_input_dtype)
        yield
        torch.set_default_dtype(default_dtype)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type when tensors get created during the module's
        forward.

        See: :meth:`torch.set_default_tensor_type`
        """
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self._desired_input_dtype)
        yield
        torch.set_default_dtype(default_dtype)
