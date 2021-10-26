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
import functools
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Union

import torch
from torch import nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device


def _do_nothing_closure() -> None:
    return None


class _LiteOptimizer:
    def __init__(self, optimizer: Optimizer, accelerator: Accelerator) -> None:
        """LiteOptimizer is a thin wrapper around the :class:`~torch.optim.Optimizer` that delegates the optimizer
        step calls to the accelerator/strategy plugin.

        The underlying wrapped optimizer object can be accessed via the property :attr:`optimizer`.

        Args:
            optimizer: The optimizer to wrap
            accelerator: Reference to the accelerator for handling the optimizer step
        """
        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k not in ("step", "__del__")}
        self.__class__ = type("Lite" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})
        self._optimizer = optimizer
        self._accelerator = accelerator

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def state(self) -> Dict[str, torch.Tensor]:
        return self._optimizer.state

    @state.setter
    def state(self, state: Dict[str, torch.Tensor]) -> None:
        self._optimizer.state = state

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._optimizer.defaults

    @defaults.setter
    def defaults(self, defaults: Dict[str, Any]) -> None:
        self._optimizer.defaults = defaults

    @property
    def param_groups(self) -> List[Dict[str, torch.Tensor]]:
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups: List[Dict[str, torch.Tensor]]) -> None:
        self._optimizer.param_groups = param_groups

    def step(self, closure: Optional[Callable] = None) -> None:
        closure = closure or _do_nothing_closure
        self._accelerator.optimizer_step(
            self._optimizer,
            opt_idx=0,
            lambda_closure=closure,
            model=self._accelerator.model,
        )

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self._optimizer.zero_grad(*args, **kwargs)


class _LiteModule(nn.Module):
    # TODO: Pass in the precision plugin instead of accelerator
    def __init__(self, module: nn.Module, accelerator: Accelerator) -> None:
        """The LiteModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            module: The module to wrap
            accelerator: Reference to the accelerator for handling precision context
        """
        super().__init__()
        self._module = module
        self._accelerator = accelerator

        if isinstance(self._module, LightningModule):
            self.training_step = self._apply_fn(self._module.training_step)
            self.validation_step = self._apply_fn(self._module.validation_step)
            self.test_step = self._apply_fn(self._module.test_step)
            self.predict_step = self._apply_fn(self._module.predict_step)
        else:
            self.forward = self._apply_fn(self._module.forward)

    def __getattr__(self, name: str) -> Any:
        if "_modules" in self.__dict__:
            # nn.Module properly instantiated.
            if name in self.__dict__:
                return self.__dict__[name]
            elif name == "_module":
                return self.__dict__["_modules"]["_module"]
            return self._module.__getattr__(name)
        return super().__getattr__(name)

    @property
    def module(self) -> nn.Module:
        return self._module

    def _apply_fn(self, func: Callable) -> Callable:
        precision_plugin = self._accelerator.precision_plugin
        precision_to_type = {
            "mixed": torch.float16,
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }
        # TODO (@awaelchli): let the precision plugin handle the conversion
        to_type = precision_to_type[precision_plugin.precision]

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            args, kwargs = apply_to_collection([args, kwargs], function=lambda t: t.to(to_type), dtype=Tensor)

            with precision_plugin.forward_context():
                output = func(*args, **kwargs)

            output = apply_to_collection(output, function=lambda t: t.to(torch.get_default_dtype()), dtype=Tensor)
            return output

        return wrapper


class _LiteDataLoader(DataLoader):
    def __init__(self, device: Optional[torch.device] = None, **dl_kwargs: Any) -> None:
        """The LiteDataLoader is an extension of the PyTorch :class:`~torch.utils.data.DataLoader` that adds
        additional features such as moving the data to the device automatically.

        Args:
            device: The device to which the data should be moved. By default the device is `None` and no data
                transfers will be made (identical behavior as :class:`~torch.utils.data.DataLoader`).
            **dl_kwargs: Accepts all arguments that the PyTorch :class:`~torch.utils.data.DataLoader` accepts.
        """
        super().__init__(**dl_kwargs)
        self._device = device

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def __iter__(self) -> Union[Iterator[Any], Generator[Any, None, None]]:  # type: ignore[override]
        iterator = super().__iter__()
        if self._device is None:
            return iterator

        for item in iterator:
            yield move_data_to_device(item, self._device)
