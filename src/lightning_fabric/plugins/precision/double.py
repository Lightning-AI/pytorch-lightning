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
from typing import Generator

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import Literal

from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor


class DoublePrecision(Precision):
    """Plugin for training with double (``torch.float64``) precision."""

    precision: Literal["64"] = "64"

    def convert_module(self, module: Module) -> Module:
        return module.double()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type.

        See: :meth:`torch.set_default_tensor_type`
        """
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        yield
        torch.set_default_dtype(default_dtype)

    def convert_input(self, data: Tensor) -> Tensor:
        return _convert_fp_tensor(data, torch.double)
