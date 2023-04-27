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
from lightning_utilities import WarningCache
from lightning_utilities.core.apply_func import apply_to_collection
from torch import nn as nn
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lightning.fabric.plugins import Precision
from lightning.fabric.strategies import Strategy
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.utilities.data import _set_sampler_epoch
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.fabric.utilities.types import Optimizable
from lightning.fabric.utilities.warnings import PossibleUserWarning

warning_cache = WarningCache()
T_destination = TypeVar("T_destination", bound=Dict[str, Any])
_LIGHTNING_MODULE_STEP_METHODS = ("training_step", "validation_step", "test_step", "predict_step")


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
        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k not in ("state_dict", "step", "__del__")}
        self.__class__ = type("Fabric" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})
        self._optimizer = optimizer
        self._strategy = strategy

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def state_dict(self) -> Dict[str, Tensor]:
        return self._strategy.get_optimizer_state(self.optimizer)

    def step(self, closure: Optional[Callable] = None) -> Any:
        kwargs = {"closure": closure} if closure is not None else {}
        if hasattr(self._strategy, "model") and isinstance(self._strategy.model, Optimizable):
            # only DeepSpeed defines this
            optimizer = self._strategy.model
        else:
            optimizer = self.optimizer
        return self._strategy.optimizer_step(
            optimizer,
            **kwargs,
        )


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
                :meth:`lightning.fabric.fabric.Fabric.setup` method. This is needed when attribute lookup
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
        args, kwargs = self._precision.convert_input((args, kwargs))

        with self._precision.forward_context():
            output = self._forward_module(*args, **kwargs)

        output = self._precision.convert_output(output)
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

    def _redirection_through_forward(self, method_name: str) -> Callable:
        assert method_name != "forward"
        original_forward = self._original_module.forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            self._original_module.forward = original_forward
            # Call the actual method e.g. `.training_step(...)`
            method = getattr(self._original_module, method_name)
            return method(*args, **kwargs)

        # We make the caller "unknowingly" send their arguments through the forward_module's `__call__`.
        # We expect that the `forward_module` will eventually call `original_module.forward`, which we
        # have patched to redirect back to `original_module.method_name()`.
        def call_forward_module(*args: Any, **kwargs: Any) -> Any:
            # Patch the original_module's forward so we can redirect the arguments back to the real method
            self._original_module.forward = wrapped_forward
            return self.forward(*args, **kwargs)

        return call_forward_module

    def _validate_method_access(self, name: str, attribute: Any) -> None:
        if inspect.ismethod(attribute) and self._forward_module != self._original_module:
            warning_cache.warn(
                f"You are calling the method `{type(self._original_module).__name__}.{name}()` from outside the"
                " model. This will bypass the wrapper from the strategy and result in incorrect behavior in"
                f" `.backward()`. You should pass your inputs through `{type(self._original_module)}.forward()`.",
                category=PossibleUserWarning,
            )

    def __getattr__(self, item: Any) -> Any:
        if item in _LIGHTNING_MODULE_STEP_METHODS and self._forward_module != self._original_module:
            # Special support for `LightningModule`, to prevent bypassing DDP's forward
            return self._redirection_through_forward(item)

        try:
            # __getattr__ gets called as a last resort if the attribute does not exist
            # call nn.Module's implementation first
            return super().__getattr__(item)
        except AttributeError:
            # If the attribute is not available on the _FabricModule wrapper, redirect to the wrapped nn.Module
            original_module = super().__getattr__("_original_module")
            attr = getattr(original_module, item)
            self._validate_method_access(item, attr)
            return attr


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
        # Without setting the epoch, the distributed sampler would return the same indices every time, even when
        # shuffling is enabled. In PyTorch, the user would normally have to call `.set_epoch()` on the sampler.
        # In Fabric, we take care of this boilerplate code.
        _set_sampler_epoch(self._dataloader, self._num_iter_calls)
        self._num_iter_calls += 1

        if self._device is None:
            yield from iter(self._dataloader)
        else:
            for item in self._dataloader:
                yield move_data_to_device(item, self._device)


def _unwrap_objects(collection: Any) -> Any:
    def _unwrap(
        obj: Union[_FabricModule, _FabricOptimizer, _FabricDataLoader]
    ) -> Union[nn.Module, Optimizer, DataLoader]:
        if isinstance(obj, _FabricModule):
            return obj._forward_module
        if isinstance(obj, _FabricOptimizer):
            return obj.optimizer
        if isinstance(obj, _FabricDataLoader):
            return obj._dataloader
        return obj

    return apply_to_collection(collection, dtype=(_FabricModule, _FabricOptimizer, _FabricDataLoader), function=_unwrap)


def is_wrapped(obj: object) -> bool:
    """Checks if an object was set up by Fabric.

    A :class:`~torch.nn.Module` may be wrapped by a :class:`_FabricModule`, a :class:`~torch.optim.Optimizer`
    may be wrapped by a :class:`_FabricOptimizer`, or a :class:`~torch.utils.data.DataLoader` may be wrapped by
    :class:`_FabricDataLoader`.

    Args:
        obj: The object to test.
    """
    return isinstance(obj, (_FabricModule, _FabricOptimizer, _FabricDataLoader))
