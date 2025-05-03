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
from collections.abc import Generator, Iterator, Mapping
from copy import deepcopy
from functools import partial, wraps
from types import MethodType
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
    overload,
)

import torch
from lightning_utilities import is_overridden
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch import nn as nn
from torch._dynamo import OptimizedModule
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from lightning.fabric.plugins import Precision
from lightning.fabric.strategies import Strategy
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.utilities.data import _set_sampler_epoch
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.fabric.utilities.types import Optimizable

T_destination = TypeVar("T_destination", bound=dict[str, Any])
_LIGHTNING_MODULE_STEP_METHODS = ("training_step", "validation_step", "test_step", "predict_step")

_in_fabric_backward: bool = False


class _FabricOptimizer:
    def __init__(self, optimizer: Optimizer, strategy: Strategy, callbacks: Optional[list[Callable]] = None) -> None:
        """FabricOptimizer is a thin wrapper around the :class:`~torch.optim.Optimizer` that delegates the optimizer
        step calls to the strategy.

        The underlying wrapped optimizer object can be accessed via the property :attr:`optimizer`.

        Args:
            optimizer: The optimizer to wrap
            strategy: Reference to the strategy for handling the optimizer step

        """
        self._optimizer = optimizer
        self._strategy = strategy
        self._callbacks = callbacks or []
        # imitate the class of the wrapped object to make isinstance checks work
        self.__class__ = type("Fabric" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def state_dict(self) -> dict[str, Tensor]:
        return self._strategy.get_optimizer_state(self.optimizer)

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def step(self, closure: Optional[Callable] = None) -> Any:
        kwargs = {"closure": closure} if closure is not None else {}
        if hasattr(self._strategy, "model") and isinstance(self._strategy.model, Optimizable):
            # only DeepSpeed defines this
            optimizer = self._strategy.model
        else:
            optimizer = self.optimizer
        output = self._strategy.optimizer_step(
            optimizer,
            **kwargs,
        )
        for callback in self._callbacks:
            hook = getattr(callback, "on_after_optimizer_step", None)
            if callable(hook):
                hook(strategy=self._strategy, optimizer=optimizer)
        return output

    def __getattr__(self, item: Any) -> Any:
        return getattr(self._optimizer, item)


class _FabricModule(_DeviceDtypeModuleMixin):
    def __init__(
        self, forward_module: nn.Module, strategy: Strategy, original_module: Optional[nn.Module] = None
    ) -> None:
        """The FabricModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            forward_module: The module to wrap the ``forward`` method on.
            strategy: Reference to the strategy for handling precision etc.
            original_module: The original, unmodified module as passed into the
                :meth:`lightning.fabric.fabric.Fabric.setup` method. This is needed when attribute lookup
                on this wrapper should pass through to the original module.

        """
        super().__init__()
        self._forward_module = forward_module
        self._original_module = original_module or forward_module
        self._strategy = strategy
        self._forward_methods = set(_LIGHTNING_MODULE_STEP_METHODS)
        self._fabric_module_initialized = True

    @property
    def module(self) -> nn.Module:
        return self._original_module or self._forward_module

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward method."""
        precision = self._strategy.precision
        args, kwargs = precision.convert_input((args, kwargs))

        with precision.forward_context():
            output = self._forward_module(*args, **kwargs)

        output = precision.convert_output(output)

        apply_to_collection(output, dtype=Tensor, function=self._register_backward_hook)
        return output

    @overload
    def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination: ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> dict[str, Any]: ...

    @override
    def state_dict(
        self, destination: Optional[T_destination] = None, prefix: str = "", keep_vars: bool = False
    ) -> Optional[dict[str, Any]]:
        return self._original_module.state_dict(
            destination=destination,  # type: ignore[type-var]
            prefix=prefix,
            keep_vars=keep_vars,
        )

    @override
    def load_state_dict(  # type: ignore[override]
        self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs: Any
    ) -> _IncompatibleKeys:
        return self._original_module.load_state_dict(state_dict=state_dict, strict=strict, **kwargs)

    def mark_forward_method(self, method: Union[MethodType, str]) -> None:
        """Mark a method as a 'forward' method to prevent it bypassing the strategy wrapper (e.g., DDP)."""
        if not isinstance(method, (MethodType, str)):
            raise TypeError(f"Expected a method or a string, but got: {type(method).__name__}")
        name = method if isinstance(method, str) else method.__name__
        if name == "forward":
            raise ValueError("You cannot mark the forward method itself as a forward method.")
        if not isinstance(getattr(self._original_module, name, None), MethodType):
            raise AttributeError(
                f"You marked '{name}' as a forward method, but `{type(self._original_module).__name__}.{name}` does not"
                f" exist or is not a method."
            )
        self._forward_methods.add(name)

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
            # Patch the original_module's forward, so we can redirect the arguments back to the real method
            self._original_module.forward = wrapped_forward
            return self.forward(*args, **kwargs)

        return call_forward_module

    def _wrap_method_with_module_call_tracker(self, method: Callable, name: str) -> Callable:
        """Tracks whether any submodule in ``self._original_module`` was called during the execution of ``method`` by
        registering forward hooks on all submodules."""
        module_called = False

        def hook(*_: Any, **__: Any) -> None:
            nonlocal module_called
            module_called = True

        @wraps(method)
        def _wrapped_method(*args: Any, **kwargs: Any) -> Any:
            handles = []
            for module in self._original_module.modules():
                handles.append(module.register_forward_hook(hook))

            output = method(*args, **kwargs)

            if module_called:
                raise RuntimeError(
                    f"You are calling the method `{type(self._original_module).__name__}.{name}()` from outside the"
                    " model. To avoid issues with the currently selected strategy, explicitly mark it as a"
                    f" forward method with `fabric_model.mark_forward_method({name!r})` after `fabric.setup()`."
                )
            for handle in handles:
                handle.remove()
            return output

        return _wrapped_method

    def _register_backward_hook(self, tensor: Tensor) -> Tensor:
        if not tensor.requires_grad:
            return tensor

        strategy_requires = is_overridden("backward", self._strategy, parent=Strategy)
        precision_requires = any(
            is_overridden(method, self._strategy.precision, parent=Precision)
            for method in ("pre_backward", "backward", "post_backward")
        )
        hook = partial(_backward_hook, (strategy_requires or precision_requires))
        tensor.register_hook(hook)
        return tensor

    @override
    def __getattr__(self, item: Any) -> Any:
        if (
            item != "_forward_methods"
            and item in self._forward_methods
            and self._forward_module != self._original_module
        ):
            # Special support for methods marked by `mark_forward_method` to prevent bypassing DDP's forward
            return self._redirection_through_forward(item)

        try:
            # __getattr__ gets called as a last resort if the attribute does not exist
            # call nn.Module's implementation first
            return super().__getattr__(item)
        except AttributeError:
            # If the attribute is not available on the _FabricModule wrapper, redirect to the wrapped nn.Module
            original_module = super().__getattr__("_original_module")
            attr = getattr(original_module, item)

            if inspect.ismethod(attr) and self._forward_module != self._original_module:
                attr = self._wrap_method_with_module_call_tracker(attr, item)
            return attr

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        if not getattr(self, "_fabric_module_initialized", False):
            super().__setattr__(name, value)
            return

        # Get the _original_module attribute
        original_module = self._original_module
        original_has_attr = hasattr(original_module, name)
        # Can't use super().__getattr__ because nn.Module only checks _parameters, _buffers, and _modules
        # Can't use self.__getattr__ because it would pass through to the original module
        fabric_has_attr = name in dir(self)

        if not (original_has_attr or fabric_has_attr):
            setattr(original_module, name, value)
            return

        # The original module can also inherit from _DeviceDtypeModuleMixin,
        # in this case, both the Fabric module and original module have attributes like _dtype
        # set attribute on both
        if original_has_attr:
            setattr(original_module, name, value)

        if fabric_has_attr:
            super().__setattr__(name, value)


class _FabricDataLoader:
    def __init__(self, dataloader: DataLoader, device: Optional[torch.device] = None) -> None:
        """The FabricDataLoader is a wrapper for the :class:`~torch.utils.data.DataLoader`. It moves the data to the
        device automatically if the device is specified.

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
        obj: Union[_FabricModule, _FabricOptimizer, _FabricDataLoader],
    ) -> Union[nn.Module, Optimizer, DataLoader]:
        if isinstance(unwrapped := _unwrap_compiled(obj)[0], _FabricModule):
            return _unwrap_compiled(unwrapped._forward_module)[0]
        if isinstance(obj, _FabricOptimizer):
            return obj.optimizer
        if isinstance(obj, _FabricDataLoader):
            return obj._dataloader
        return obj

    types = [_FabricModule, _FabricOptimizer, _FabricDataLoader]
    types.append(OptimizedModule)

    return apply_to_collection(collection, dtype=tuple(types), function=_unwrap)


def _unwrap_compiled(obj: Union[Any, OptimizedModule]) -> tuple[Union[Any, nn.Module], Optional[dict[str, Any]]]:
    """Removes the :class:`torch._dynamo.OptimizedModule` around the object if it is wrapped.

    Use this function before instance checks against e.g. :class:`_FabricModule`.

    """
    if isinstance(obj, OptimizedModule):
        if (compile_kwargs := getattr(obj, "_compile_kwargs", None)) is None:
            raise RuntimeError(
                "Failed to determine the arguments that were used to compile the module. Make sure to import"
                " lightning before `torch.compile` is used."
            )
        return obj._orig_mod, compile_kwargs
    return obj, None


def _to_compiled(module: nn.Module, compile_kwargs: dict[str, Any]) -> OptimizedModule:
    return torch.compile(module, **compile_kwargs)  # type: ignore[return-value]


def _backward_hook(requires_backward: bool, *_: Any) -> None:
    if requires_backward and not _in_fabric_backward:
        raise RuntimeError(
            "The current strategy and precision selection requires you to call `fabric.backward(loss)`"
            " instead of `loss.backward()`."
        )


def is_wrapped(obj: object) -> bool:
    """Checks if an object was set up by Fabric.

    A :class:`~torch.nn.Module` may be wrapped by a :class:`_FabricModule`, a :class:`~torch.optim.Optimizer`
    may be wrapped by a :class:`_FabricOptimizer`, or a :class:`~torch.utils.data.DataLoader` may be wrapped by
    :class:`_FabricDataLoader`.

    Args:
        obj: The object to test.

    """
    obj, _ = _unwrap_compiled(obj)
    return isinstance(obj, (_FabricModule, _FabricOptimizer, _FabricDataLoader))


def _capture_compile_kwargs(compile_fn: Callable) -> Callable:
    """Wraps the ``torch.compile`` function and captures the compile arguments.

    We extract the compile arguments so that we can reapply ``torch.compile`` in ``Fabric.setup()`` with the
    same arguments as the user passed to the original call. The arguments get stored in a dictionary
    ``_compile_kwargs`` on the returned compiled module.

    """
    # Limitation: Currently, the global compile config does not get captured on a per-model basis.
    # PyTorch will resolve this in the future: https://github.com/pytorch/pytorch/issues/116575

    @wraps(compile_fn)
    def _capture(*args: Any, **kwargs: Any) -> Any:
        if not args or not isinstance(args[0], nn.Module):
            # either torch.compile is being applied as a decorator or we're compiling something else
            return compile_fn(*args, **kwargs)

        model = args[0]
        compiled_model = compile_fn(model, **kwargs)
        compiled_model._compile_kwargs = deepcopy(kwargs)
        return compiled_model

    return _capture


torch.compile = _capture_compile_kwargs(torch.compile)
