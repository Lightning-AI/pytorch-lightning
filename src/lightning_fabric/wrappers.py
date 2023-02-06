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
import inspect
from typing import Any, Callable, Dict, Generator, Iterator, Mapping, Optional, overload, TypeVar, Union

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import nn as nn
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lightning_fabric.plugins import Precision
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor
from lightning_fabric.strategies import Strategy
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_fabric.utilities.types import Optimizable

T_destination = TypeVar("T_destination", bound=Dict[str, Any])


class _FabricOptimizer:
    def __init__(self, optimizer: Optimizer, strategy: Strategy) -> None:
        """FabricOptimizer is a thin wrapper around the :class:`~torch.optim.Optimizer` that delegates the
        optimizer step calls to the strategy plugin.

        The underlying wrapped optimizer object can be accessed via the property :attr:`optimizer`.

        Args:
            optimizer: The optimizer to wrap
            strategy: Reference to the strategy for handling the optimizer step
        """
        # `__del__` is skipped in case the optimizer has implemented custom destructor logic which we would
        # not want to call on destruction of the `_FabricOptimizer
        self.__dict__ = {
            k: v for k, v in optimizer.__dict__.items() if k not in ("state_dict", "step", "zero_grad", "__del__")
        }
        self.__class__ = type("Fabric" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})
        self._optimizer = optimizer
        self._strategy = strategy

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def state_dict(self) -> Dict[str, Tensor]:
        return self._strategy.get_optimizer_state(self.optimizer)

    def step(self, closure: Optional[Callable] = None) -> Any:
        kwargs = dict(closure=closure) if closure is not None else {}
        if hasattr(self._strategy, "model") and isinstance(self._strategy.model, Optimizable):
            # only DeepSpeed defines this
            optimizer = self._strategy.model
        else:
            optimizer = self.optimizer
        return self._strategy.optimizer_step(
            optimizer,
            **kwargs,
        )

    def zero_grad(self, **kwargs: Any) -> None:
        kwargs = _process_optimizer_zero_grad_kwargs(self.optimizer, kwargs)
        self.optimizer.zero_grad(**kwargs)


class _FabricModule(_DeviceDtypeModuleMixin):
    def __init__(
        self, forward_module: nn.Module, precision: Precision, original_module: Optional[nn.Module] = None
    ) -> None:
        """The FabricModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            forward_module: The module to wrap the ``forward`` method on.
            precision: Reference to the precision plugin for handling precision context
            original_module: The original, unmodified module as passed into the
                :meth:`lightning_fabric.fabric.Fabric.setup` method. This is needed when attribute lookup
                on this wrapper should pass through to the original module.
        """
        super().__init__()
        self._forward_module = forward_module
        self._original_module = original_module or forward_module
        self._precision = precision

    @property
    def module(self) -> nn.Module:
        return self._original_module or self._forward_module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward
        method."""
        args, kwargs = apply_to_collection([args, kwargs], function=self._precision.convert_input, dtype=Tensor)

        with self._precision.forward_context():
            output = self._forward_module(*args, **kwargs)

        output = apply_to_collection(
            output, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype()
        )
        return output

    @overload
    def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
        ...

    def state_dict(
        self, destination: Optional[T_destination] = None, prefix: str = "", keep_vars: bool = False
    ) -> Optional[Dict[str, Any]]:
        return self._original_module.state_dict(
            destination=destination,  # type: ignore[type-var]
            prefix=prefix,
            keep_vars=keep_vars,
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> _IncompatibleKeys:
        return self._original_module.load_state_dict(state_dict=state_dict, strict=strict)

    def __getattr__(self, item: Any) -> Any:
        try:
            # __getattr__ gets called as a last resort if the attribute does not exist
            # call nn.Module's implementation first
            return super().__getattr__(item)
        except AttributeError:
            # If the attribute is not available on the _FabricModule wrapper, redirect to the wrapped nn.Module
            original_module = super().__getattr__("_original_module")
            return getattr(original_module, item)


class _FabricDataLoader:
    def __init__(self, dataloader: DataLoader, device: Optional[torch.device] = None) -> None:
        """The FabricDataLoader is a wrapper for the :class:`~torch.utils.data.DataLoader`. It moves the data to
        the device automatically if the device is specified.

        Args:
            dataloader: The dataloader to wrap
            device: The device to which the data should be moved. By default the device is `None` and no data
                transfers will be made (identical behavior as :class:`~torch.utils.data.DataLoader`).
        """
        self.__dict__.update(dataloader.__dict__)
        self._dataloader = dataloader
        self._device = device
        self._num_iter_calls = 0

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Union[Iterator[Any], Generator[Any, None, None]]:
        if hasattr(self._dataloader.sampler, "set_epoch"):
            # Without setting the epoch, the distributed sampler would return the same indices every time, even when
            # shuffling is enabled. In PyTorch, the user would normally have to call `.set_epoch()` on the sampler.
            # In Lite, we take care of this boilerplate code.
            self._dataloader.sampler.set_epoch(self._num_iter_calls)
        self._num_iter_calls += 1

        iterator = iter(self._dataloader)
        if self._device is None:
            yield from iterator
            return

        for item in iterator:
            yield move_data_to_device(item, self._device)


def _process_optimizer_zero_grad_kwargs(optimizer: Optimizer, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if "set_to_none" in kwargs and "set_grads_to_None" in inspect.signature(optimizer.zero_grad).parameters:
        # Some optimizers out there, for example DeepSpeedZeroOptimizer, use a different name than PyTorch
        kwargs["set_grads_to_None"] = kwargs.pop("set_to_none")
    return kwargs
