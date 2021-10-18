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
from typing import Any, Callable, Generator, Iterator, Optional, Union

import torch
from torch import nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device


class _LiteOptimizer:
    def __init__(self, optimizer: Optimizer, accelerator: Accelerator) -> None:
        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k not in ("step", "__del__")}
        self.__class__ = type("Lite" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})
        self._optimizer = optimizer
        self._accelerator = accelerator

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def state(self):
        return self._optimizer.state

    @state.setter
    def state(self, state):
        self._optimizer.state = state

    @property
    def defaults(self):
        return self._optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self._optimizer.defaults = defaults

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self._optimizer.param_groups = param_groups

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
        with self._accelerator.precision_plugin.forward_context():
            output = self.module.forward(*args, **kwargs)

        output = apply_to_collection(output, function=lambda t: t.to(torch.get_default_dtype()), dtype=Tensor)
        return output


class _LiteDataLoader(DataLoader):
    def __init__(self, device: Optional[torch.device] = None, **dl_kwargs: Any) -> None:
        super().__init__(**dl_kwargs)
        self._device = device

    def __iter__(self) -> Union[Iterator[Any], Generator[Any, None, None]]:
        iterator = super().__iter__()
        if self._device is None:
            return iterator

        for item in iterator:
            yield move_data_to_device(item, self._device)
