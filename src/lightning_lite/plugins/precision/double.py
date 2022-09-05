# Copyright The PyTorch Lightning team.
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
from torch import FloatTensor

from lightning_lite.plugins import PrecisionPlugin


class DoublePrecisionPlugin(PrecisionPlugin):
    """Plugin for training with double (``torch.float64``) precision."""

    precision: int = 64

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type.

        See: :meth:`torch.set_default_tensor_type`
        """
        torch.set_default_tensor_type(torch.DoubleTensor)
        yield
        torch.set_default_tensor_type(FloatTensor)
