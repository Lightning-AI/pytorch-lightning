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
from __future__ import annotations

import os
from typing import Any, Literal

import torch.nn as nn
from torch.optim import Optimizer

from lightning.pytorch.plugins.precision import XLAPrecisionPlugin


class XLABf16PrecisionPlugin(XLAPrecisionPlugin):
    """Plugin that enables mixed bf16 with XLA."""

    precision: Literal["bf16-mixed"] = "bf16-mixed"

    def connect(
        self, model: nn.Module, optimizers: list[Optimizer], lr_schedulers: list[Any]
    ) -> tuple[nn.Module, list[Optimizer], list[Any]]:
        os.environ["XLA_USE_BF16"] = "1"
        return super().connect(model=model, optimizers=optimizers, lr_schedulers=lr_schedulers)

    def teardown(self) -> None:
        os.environ.pop("XLA_USE_BF16", None)
