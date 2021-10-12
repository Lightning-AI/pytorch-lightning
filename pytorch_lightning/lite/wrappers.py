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

from torch import nn as nn
from torch.optim import Optimizer

from pytorch_lightning.accelerators import Accelerator


class _LiteOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, accelerator: Accelerator):
        super().__init__(params=optimizer.param_groups, defaults=optimizer.defaults)
        self.optimizer = optimizer
        self._accelerator = accelerator

    def step(self, closure=None, **kwargs: Any):
        print("running automated step")
        output = self._accelerator.run_optimizer_step(
            self.optimizer,
            lambda_closure=closure,
            **kwargs,
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
        return output
