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
from typing import Any, Callable, Dict, Generator, Iterator, Optional, Union

import torch
from torch import nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin
from pytorch_lightning.plugins import PrecisionPlugin
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device


def _do_nothing_closure() -> None:
    return None


class _LiteOptimizer:
    def __init__(self, optimizer: Optimizer, strategy: Strategy) -> None:
        """LiteOptimizer is a thin wrapper around the :class:`~torch.optim.Optimizer` that delegates the optimizer
        step calls to the strategy plugin.

        The underlying wrapped optimizer object can be accessed via the property :attr:`optimizer`.

        Args:
            optimizer: The optimizer to wrap
            strategy: Reference to the strategy for handling the optimizer step
        """
        # `__del__` is skipped in case the optimizer has implemented custom destructor logic which we would
        # not want to call on destruction of the `_LiteOptimizer
        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k not in ("state_dict", "step", "__del__")}
        self.__class__ = type("Lite" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})
        self._optimizer = optimizer
        self._strategy = strategy

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def state_dict(self) -> Dict[str, Tensor]:
        return self._strategy.optimizer_state(self.optimizer)

    def step(self, closure: Optional[Callable] = None) -> Any:
        closure = closure or _do_nothing_closure
        return self._strategy.optimizer_step(
            self.optimizer,
            opt_idx=0,
            closure=closure,
            model=self._strategy.model,
        )


class _LiteModule(DeviceDtypeModuleMixin):
    def __init__(
        self, forward_module: nn.Module, precision_plugin: PrecisionPlugin, original_module: Optional[nn.Module] = None
    ) -> None:
        """The LiteModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            forward_module: The module to wrap the ``forward`` method on.
            precision_plugin: Reference to the precision plugin for handling precision context
            original_module: The original, unmodified module as passed into the
                :meth:`pytorch_lightning.lite.lite.LightningLite.setup` method. This is needed when attribute lookup
                on this wrapper should pass through to the original module.
        """
        super().__init__()
        self._forward_module = forward_module
        self._original_module = original_module or forward_module
        self._precision_plugin = precision_plugin

    @property
    def module(self) -> nn.Module:
        return self._original_module or self._forward_module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward
        method."""
        precision = self._precision_plugin.precision
        precision_to_type = {
            "bf16": torch.bfloat16,
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }
        # TODO (@awaelchli): let the precision plugin handle the conversion
        to_type = precision_to_type[precision]

        def _convert_float_tensor(t: Tensor) -> Tensor:
            return t.to(to_type) if torch.is_floating_point(t) else t

        args, kwargs = apply_to_collection([args, kwargs], function=_convert_float_tensor, dtype=Tensor)

        with self._precision_plugin.forward_context():
            output = self._forward_module(*args, **kwargs)

        to_type = torch.get_default_dtype()
        output = apply_to_collection(output, function=_convert_float_tensor, dtype=Tensor)
        return output

    def __getattr__(self, item: Any) -> Any:
        try:
            # __getattr__ gets called as a last resort if the attribute does not exist
            # call nn.Module's implementation first
            return super().__getattr__(item)
        except AttributeError:
            # If the attribute is not available on the _LiteModule wrapper, redirect to the wrapped nn.Module
            original_module = super().__getattr__("_original_module")
            return getattr(original_module, item)


class _LiteDataLoader:
    def __init__(self, dataloader: DataLoader, device: Optional[torch.device] = None) -> None:
        """The LiteDataLoader is a wrapper for the :class:`~torch.utils.data.DataLoader`. It moves the data to the
        device automatically if the device is specified.

        Args:
            dataloader: The dataloader to wrap
            device: The device to which the data should be moved. By default the device is `None` and no data
                transfers will be made (identical behavior as :class:`~torch.utils.data.DataLoader`).
        """
        self.__dict__.update(dataloader.__dict__)
        self._dataloader = dataloader
        self._device = device

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Union[Iterator[Any], Generator[Any, None, None]]:
        iterator = iter(self._dataloader)
        if self._device is None:
            yield from iterator
            return

        for item in iterator:
            yield move_data_to_device(item, self._device)
