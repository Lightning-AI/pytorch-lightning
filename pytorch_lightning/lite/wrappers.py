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
from typing import Any, Callable, Optional

import torch
from torch import nn as nn, Tensor
from torch.optim import Optimizer

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.utilities.apply_func import apply_to_collection


class _LiteOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, accelerator: Accelerator) -> None:
        super().__init__(params=optimizer.param_groups, defaults=optimizer.defaults)  # type: ignore[call-arg]
        self.__dict__ = optimizer.__dict__
        self._optimizer = optimizer
        self._accelerator = accelerator

    def step(self, closure: Optional[Callable] = None) -> None:
        self._accelerator.optimizer_step(
            self._optimizer,
            lambda_closure=closure,
            model=None,
        )


class _LiteModule(nn.Module):
    def __init__(self, module: nn.Module, accelerator: Accelerator) -> None:
        super().__init__()
        self._module = module
        self._accelerator = accelerator

    @property
    def module(self) -> nn.Module:
        return self._module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with self._accelerator.forward_context():
            output = self.module.forward(*args, **kwargs)

        output = apply_to_collection(output, function=lambda t: t.to(torch.get_default_dtype()), dtype=Tensor)
        return output
