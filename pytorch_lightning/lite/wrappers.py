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

from typing import Any

import torch
from torch import nn as nn, Tensor
from torch.optim import Optimizer

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.utilities.apply_func import apply_to_collection


class _LiteOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, accelerator: Accelerator):
        super().__init__(params=optimizer.param_groups, defaults=optimizer.defaults)
        self.optimizer = optimizer
        self._accelerator = accelerator

    def step(self, closure=None):
        print("running automated step")
        output = self._accelerator.optimizer_step(
            self.optimizer,
            lambda_closure=closure,
            model=None,
        )
        return output


class _LiteModel(nn.Module):
    def __init__(self, module: nn.Module, accelerator: Accelerator):
        super().__init__()
        self._module = module
        self._accelerator = accelerator

    @property
    def module(self):
        return self._module

    def forward(self, *args, **kwargs):
        with self._accelerator.forward_context():
            output = self.module.forward(*args, **kwargs)

        output = apply_to_collection(output, function=lambda t: t.to(torch.get_default_dtype()), dtype=Tensor)
        return output
