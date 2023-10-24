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
import weakref
from functools import wraps
from numbers import Number
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch import nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils._pytree import map_only, tree_flatten, tree_map_only, tree_unflatten
from torch.utils.data import DataLoader

from lightning.fabric.strategies import DeepSpeedStrategy, Strategy
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.utilities.data import _set_sampler_epoch
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_2
from lightning.fabric.utilities.types import Optimizable

T_destination = TypeVar("T_destination", bound=Dict[str, Any])
_LIGHTNING_MODULE_STEP_METHODS = ("training_step", "validation_step", "test_step", "predict_step")


class _FabricOptimizer:
    def __init__(self, optimizer: Optimizer, strategy: Strategy, callbacks: Optional[List[Callable]] = None) -> None:
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

    def state_dict(self) -> Dict[str, Tensor]:
        return self._strategy.get_optimizer_state(self.optimizer)

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
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
            strategy: Reference to the strategy.
            original_module: The original, unmodified module as passed into the
                :meth:`lightning.fabric.fabric.Fabric.setup` method. This is needed when attribute lookup
                on this wrapper should pass through to the original module.

        """
        super().__init__()
        self._forward_module = forward_module
        self._original_module = original_module or forward_module
        self._strategy = strategy
        self._fabric_module_initialized = True

    @property
    def module(self) -> nn.Module:
        return self._original_module or self._forward_module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward method."""
        precision = self._strategy.precision
        args, kwargs = precision.convert_input((args, kwargs))

        with precision.forward_context():
            output = self._forward_module(*args, **kwargs)
            if isinstance(output, torch.Tensor) and torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
                output = _BackwardTensor._from_tensor(output, self._strategy, self._forward_module)

        output = precision.convert_output(output)
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

    def load_state_dict(  # type: ignore[override]
        self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs: Any
    ) -> _IncompatibleKeys:
        return self._original_module.load_state_dict(state_dict=state_dict, strict=strict, **kwargs)

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
                    " model. This will bypass the wrapper from the strategy and result in incorrect behavior in"
                    " `.backward()`. You should pass your inputs through `forward()`.",
                )
            for handle in handles:
                handle.remove()
            return output

        return _wrapped_method

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

            if inspect.ismethod(attr) and self._forward_module != self._original_module:
                attr = self._wrap_method_with_module_call_tracker(attr, item)
            return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if not getattr(self, "_fabric_module_initialized", False):
            super().__setattr__(name, value)
            return

        # Get the _original_module attribute
        original_module = self._original_module
        original_has_attr = hasattr(original_module, name)
        # Can't use super().__getattr__ because nn.Module only checks _parameters, _buffers, and _modules
        # Can't use self.__getattr__ because it would pass through to the original module
        fabric_has_attr = name in self.__dict__

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


class _BackwardTensor(torch.Tensor):
    """Tensor wrapper subclass that allows us to hook the strategy machinery into ``loss.backward()``.

    The implementation is inspired by:
        - https://pytorch.org/docs/main/notes/extending.html#operations-on-multiple-types-that-define-torch-function
        - :class:`~torch.distributed._tensor.api.DTensor`
        - https://github.com/albanD/subclass_zoo/blob/08dfcf9/custom_parameter.py
        - https://github.com/albanD/subclass_zoo/issues/29
        - :class:`~torch.testing._internal.two_tensor`
        - https://github.com/pytorch/pytorch/blob/ba86dfcd83/torch/testing/_internal/two_tensor.py

    """

    def __new__(cls, tensor: torch.Tensor, strategy: Strategy, module: torch.nn.Module) -> "_BackwardTensor":
        wrapper = torch.Tensor._make_wrapper_subclass(
            cls,
            size=tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )
        wrapper._tensor = tensor
        # use weak refs so that these tensors can still be used after fabric is garbage-collected
        wrapper._strategy_ref = weakref.ref(strategy) if not isinstance(strategy, weakref.ReferenceType) else strategy
        wrapper._module_ref = weakref.ref(module) if not isinstance(module, weakref.ReferenceType) else module
        return wrapper

    def backward(self, *args: Any, **kwargs: Any) -> None:
        tensor = self._to_tensor()
        strategy = self._strategy_ref()
        module = self._module_ref()

        if strategy is None or module is None:
            # fabric was gc-collected: silently fallback to ordinary backward
            tensor.backward()
            return

        if isinstance(strategy, DeepSpeedStrategy):
            # requires to attach the current `DeepSpeedEngine` for the `_FabricOptimizer.step` call.
            strategy._deepspeed_engine = module
        strategy.backward(tensor, module, *args, **kwargs)

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable,
        types: Iterable[Type],
        args: Tuple,
        kwargs: Dict,
    ) -> Any:
        flat_args, spec = tree_flatten(args)
        bwd_tensors = [a for a in flat_args if isinstance(a, _BackwardTensor)]
        assert bwd_tensors
        bwd_tensor = bwd_tensors[0]
        strategy_ref = bwd_tensor._strategy_ref
        module_ref = bwd_tensor._module_ref
        strategy = strategy_ref()
        module = module_ref()

        # sanity checking
        for other in bwd_tensors[1:]:
            if type(other._strategy_ref()) is not type(strategy):
                raise NotImplementedError(
                    "Calling `backward` on tensors that were produced by Fabric objects with different strategies"
                    f" is not supported. Found {type(strategy).__name__} and {type(other._strategy_ref()).__name__}"
                )
            if isinstance(strategy, DeepSpeedStrategy) and other._module_ref() is not module:
                raise NotImplementedError(
                    "Calling `backward` on tensors that were produced by different fabric modules is not supported."
                    f" Found {module} and {other._module_ref()}"
                )

        # this could use `tree_map_only` but it's decomposed since we already have the flat args and spec
        @map_only(_BackwardTensor)
        def unwrap(t: _BackwardTensor) -> torch.Tensor:
            return t._tensor

        unwrapped = tree_unflatten([unwrap(i) for i in flat_args], spec)

        # run the original function
        out = func(*unwrapped, **kwargs)

        # if garbage-collected, do not propagate
        if strategy is None or module is None:
            return out

        def wrap(t: torch.Tensor) -> _BackwardTensor:
            return _BackwardTensor(t, strategy_ref, module_ref)

        # propagate the references from the inputs to the outputs
        out = tree_map_only(torch.Tensor, wrap, out)

        if _TORCH_GREATER_EQUAL_2_2:
            # fix aliasing for torch.compile support with wrapper subclasses
            from torch.utils._python_dispatch import return_and_correct_aliasing

            out = return_and_correct_aliasing(func, args, kwargs, out)

        return out

    # The default `__torch_function__` will always wrap outputs into a subclass if they aren't already a subclass.
    # we don't want this in the case where the strategy/module has been garbage-collected.
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def _from_tensor(tensor: torch.Tensor, strategy: Strategy, module: torch.nn.Module) -> "_BackwardTensor":
        return _FromTensor.apply(tensor, strategy, module)

    def _to_tensor(self) -> torch.Tensor:
        return _ToTensor.apply(self)

    def __repr__(self, *, tensor_contents: Optional[str] = None) -> str:
        # avoid showing `_BackwardTensor` to the user
        return self._tensor.__repr__(tensor_contents=tensor_contents)

    def tolist(self) -> Union[Number, List]:
        # workaround to https://github.com/pytorch/pytorch/blob/v2.1.0/torch/csrc/utils/tensor_list.cpp#L43-L45
        return self._tensor.tolist()


class _FromTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        strategy: Strategy,
        module: torch.nn.Module,
    ) -> _BackwardTensor:  # type: ignore[override]
        return _BackwardTensor(tensor, strategy, module)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Union[_BackwardTensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, None, None]:  # type: ignore[override]
        if isinstance(grad_output, _BackwardTensor):
            # this wouldn't be a BackwardTensor if the strategy/module has been garbage collected
            grad_output = grad_output._to_tensor()
        return grad_output, None, None


class _ToTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: _BackwardTensor,
    ) -> torch.Tensor:  # type: ignore[override]
        ctx.strategy_ref = tensor._strategy_ref
        ctx.module_ref = tensor._module_ref
        return tensor._tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> _BackwardTensor:  # type: ignore[override]
        return _BackwardTensor(grad_output, ctx.strategy_ref, ctx.module_ref)


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
        obj: Union[_FabricModule, _FabricOptimizer, _FabricDataLoader]
    ) -> Union[nn.Module, Optimizer, DataLoader]:
        if isinstance(unwrapped := _unwrap_compiled(obj), _FabricModule):
            return unwrapped._forward_module
        if isinstance(obj, _FabricOptimizer):
            return obj.optimizer
        if isinstance(obj, _FabricDataLoader):
            return obj._dataloader
        return obj

    types = [_FabricModule, _FabricOptimizer, _FabricDataLoader]
    if _TORCH_GREATER_EQUAL_2_0:
        from torch._dynamo import OptimizedModule

        types.append(OptimizedModule)

    return apply_to_collection(collection, dtype=tuple(types), function=_unwrap)


def _unwrap_compiled(obj: Any) -> Any:
    """Removes the :class:`torch._dynamo.OptimizedModule` around the object if it is wrapped.

    Use this function before instance checks against e.g. :class:`_FabricModule`.

    """
    if not _TORCH_GREATER_EQUAL_2_0:
        return obj
    from torch._dynamo import OptimizedModule

    if isinstance(obj, OptimizedModule):
        return obj._orig_mod
    return obj


def is_wrapped(obj: object) -> bool:
    """Checks if an object was set up by Fabric.

    A :class:`~torch.nn.Module` may be wrapped by a :class:`_FabricModule`, a :class:`~torch.optim.Optimizer`
    may be wrapped by a :class:`_FabricOptimizer`, or a :class:`~torch.utils.data.DataLoader` may be wrapped by
    :class:`_FabricDataLoader`.

    Args:
        obj: The object to test.

    """
    obj = _unwrap_compiled(obj)
    return isinstance(obj, (_FabricModule, _FabricOptimizer, _FabricDataLoader))
