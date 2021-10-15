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
import contextlib
import importlib
import inspect
import threading
from contextlib import contextmanager
from itertools import chain
from typing import Callable, Dict, Generator, Iterator, Optional

import torch
from torch import nn, Tensor
from torch.nn import Module

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_META_AVAILABLE

if _TORCH_META_AVAILABLE:
    from torch._C import _DisableTorchDispatch  # type: ignore[attr-defined]

    ####################################################################
    # BELOW: TAKEN FROM https://github.com/pytorch/pytorch/pull/66317. #
    # TODO: Removed once merged and released on PyTorch side           #
    ####################################################################

    @contextlib.contextmanager
    def enable_python_mode(cls) -> Iterator[None]:
        if not hasattr(cls, "__torch_dispatch__"):
            raise ValueError("The class passed to enable_python_mode " "must have a __torch_dispatch__ classmethod")
        if not isinstance(cls, type) or not issubclass(cls, (torch.Tensor,)):
            raise ValueError("The argument passed to enable_python_mode " "must be the type of a Tensor subclass")
        torch._C._enter_python_mode(cls)
        try:
            yield
        finally:
            torch._C._exit_python_mode()

    _tls = threading.local()
    _tls.in_call = False

    @contextmanager
    def _no_dispatch() -> Iterator[None]:
        """Temporarily disables the Python dispatch mode."""
        guard = _DisableTorchDispatch()  # noqa F841
        try:
            yield
        finally:
            del guard

    def _handle_arange(func, args, kwargs):
        kwargs["device"] = torch.device("cpu")
        return torch.empty_like(func(*args, **kwargs), device="meta")

    def _handle_tril(func, args, kwargs):
        if args and isinstance(args[0], Tensor):
            return torch.empty_like(args[0], device="meta")

        return NotImplemented

    class _MetaContext(Tensor):

        _op_handlers: Dict[Callable, Callable] = {}

        @classmethod
        def _ensure_handlers_initialized(cls) -> None:
            if cls._op_handlers:
                return

            cls._op_handlers.update(
                {
                    torch.ops.aten.arange: _handle_arange,
                    torch.ops.aten.tril: _handle_tril,
                }
            )

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            cls._ensure_handlers_initialized()

            op_handler: Optional[Callable]

            try:
                op_handler = cls._op_handlers[func]
            except KeyError:
                op_handler = None

            with _no_dispatch():
                if op_handler:
                    result = op_handler(func, args, kwargs)
                    if result is not NotImplemented:
                        return result

                if "device" in kwargs:
                    kwargs["device"] = torch.device("meta")

                return func(*args, **kwargs)

    def init_meta(module_fn: Callable[..., Module], *args, **kwargs) -> Module:
        def create_instance() -> Module:
            return module_fn(*args, **kwargs)

        if _tls.in_call:
            module = create_instance()
        else:
            _tls.in_call = True
            try:
                with enable_python_mode(_MetaContext):
                    module = create_instance()
            finally:
                _tls.in_call = False

        module.materialize = create_instance  # type: ignore[assignment]

        return module

    def is_meta_init() -> bool:
        """Indicates whether the module is being instantiated by ``init_meta()``."""
        return _tls.in_call

    ####################################################################
    # ABOVE: TAKEN FROM https://github.com/pytorch/pytorch/pull/66317. #
    # TODO: Removed once merged and released on PyTorch side           #
    ####################################################################


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


def recursively_setattr(root_module: nn.Module, prefix: str, materialized_module: nn.Module):
    *path, name = prefix.split(".")
    for p in path:
        root_module = getattr(root_module, p)

    try:
        index = int(name)
        root_module[index] = materialized_module
    except ValueError:
        setattr(root_module, name, materialized_module)


def materialize_module(root_module: torch.nn.Module):
    """This utility performs an in-place operation by materialize a module and its children."""
    memo = []
    modules = list(root_module.named_modules())
    for prefix, mod in modules:
        materialize_fn = getattr(mod, "materialize", None)
        if materialize_fn:
            memo.append((prefix, materialize_fn()))
    for prefix, materialized_module in memo:
        recursively_setattr(root_module, prefix, materialized_module)


# cache to optimize the search while resetting later on.
__STORAGE_META__ = {}


def _unset_meta_device() -> None:
    """Replace all meta module by their original version."""
    if not _TORCH_META_AVAILABLE:
        raise MisconfigurationException("`init_meta` is supported from PyTorch 1.10.0")

    for mods, subclass, _ in __STORAGE_META__.values():
        for mod in mods:
            setattr(mod, subclass.__name__, subclass)


def _set_meta_device() -> None:
    """Replace all torch.nn.Module by their meta replacement."""

    if not _TORCH_META_AVAILABLE:
        raise MisconfigurationException("`init_meta` is supported from PyTorch 1.10.0")

    # Find all the nn.Module subclasses
    for subclass in get_all_subclasses(torch.nn.modules.module.Module):

        # if subclass has already been stored, use teh cache
        if str(subclass) in __STORAGE_META__:
            # reset the class import package to its rightfull state.
            mods, subclass, meta_class = __STORAGE_META__[str(subclass)]
            for mod in mods:
                setattr(mod, subclass.__name__, meta_class)
            continue

        # Create a class subclassing current `subclass` overriding its new method.
        # this will enable use to use `torch.distributed.nn.utils.init_meta` to create a `meta`
        # version of the current subclass module
        class _MetaClass(subclass):
            def __new__(cls, *args, **kwargs):
                # access the current subclass
                subclass = cls.__bases__[0]
                submodules = subclass.__module__.split(".")
                # import its package
                mod = importlib.import_module(submodules[0])
                for name in submodules[1:]:
                    mod = getattr(mod, name)

                # replace the package to its rightful form, so python instantiation
                # works as expected.
                setattr(mod, subclass.__name__, subclass)

                # create meta module
                obj = init_meta(subclass, *args, **kwargs)

                obj._materialize = obj.materialize

                # the `materialize` function need to be overridden as the same
                # toggle logic need to be used to enable proper module instantiation.
                def materialize():
                    nonlocal obj
                    setattr(mod, subclass.__name__, subclass)
                    obj = obj._materialize()
                    setattr(mod, subclass.__name__, cls)
                    return obj

                obj.materialize = materialize
                # replace the package to its meta form, so future instantation are still in the meta form.
                setattr(mod, subclass.__name__, cls)
                return obj

        def search(mod):
            out = []
            for _, obj in inspect.getmembers(mod):
                if obj == subclass:
                    out.append(mod)
            return out

        submodules = subclass.__module__.split(".")
        mod = importlib.import_module(submodules[0])

        # nn.Module class can be imported at different level and they all need to be mocked.
        # Example: torch.nn.Linear is actually torch.nn.modules.linear.Linear
        # Therefore, torch.nn.Linear, torch.nn.modules.Linear, torch.nn.modules.linear.Linear
        # needs to be replaced by the torch.nn.linear.modules.Linear _MetaClass
        out = []
        out.append(search(mod))
        for name in submodules[1:]:
            mod = getattr(mod, name)
            out.append(search(mod))

        # drop empty module
        mods = [mod for mod in chain(*out) if mod]

        # store the modules search so it doesn't have to be performed again for this class
        __STORAGE_META__[str(subclass)] = (mods, subclass, _MetaClass)

        # replace all subclass by its meta form
        for mod in mods:
            setattr(mod, subclass.__name__, _MetaClass)


@contextmanager
def init_meta_context() -> Generator:
    _set_meta_device()
    yield
    _unset_meta_device()
