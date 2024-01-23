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
import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Type, TypeVar

from lightning_utilities.core.imports import RequirementCache
from torch import nn
from typing_extensions import Concatenate, ParamSpec

import lightning.pytorch as pl

_log = logging.getLogger(__name__)


def is_overridden(method_name: str, instance: Optional[object] = None, parent: Optional[Type[object]] = None) -> bool:
    if instance is None:
        # if `self.lightning_module` was passed as instance, it can be `None`
        return False
    if parent is None:
        if isinstance(instance, pl.LightningModule):
            parent = pl.LightningModule
        elif isinstance(instance, pl.LightningDataModule):
            parent = pl.LightningDataModule
        elif isinstance(instance, pl.Callback):
            parent = pl.Callback
        if parent is None:
            _check_mixed_imports(instance)
            raise ValueError("Expected a parent")

    from lightning_utilities.core.overrides import is_overridden as _is_overridden

    return _is_overridden(method_name, instance, parent)


def get_torchvision_model(model_name: str, **kwargs: Any) -> nn.Module:
    from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

    if not _TORCHVISION_AVAILABLE:
        raise ModuleNotFoundError(str(_TORCHVISION_AVAILABLE))

    from torchvision import models

    torchvision_greater_equal_0_14 = RequirementCache("torchvision>=0.14.0")
    # TODO: deprecate this function when 0.14 is the minimum supported torchvision
    if torchvision_greater_equal_0_14:
        return models.get_model(model_name, **kwargs)
    return getattr(models, model_name)(**kwargs)


class _ModuleMode:
    """Captures the ``nn.Module.training`` (bool) mode of every submodule, and allows it to be restored later on."""

    def __init__(self) -> None:
        self.mode: Dict[str, bool] = {}

    def capture(self, module: nn.Module) -> None:
        self.mode.clear()
        for name, mod in module.named_modules():
            self.mode[name] = mod.training

    def restore(self, module: nn.Module) -> None:
        for name, mod in module.named_modules():
            if name not in self.mode:
                _log.debug(
                    f"Restoring training mode on module '{name}' not possible, it was never captured."
                    f" Is your module structure changing?"
                )
                continue
            mod.training = self.mode[name]


def _check_mixed_imports(instance: object) -> None:
    old, new = "pytorch_" + "lightning", "lightning." + "pytorch"
    klass = type(instance)
    module = klass.__module__
    if module.startswith(old) and __name__.startswith(new):
        pass
    elif module.startswith(new) and __name__.startswith(old):
        old, new = new, old
    else:
        return
    raise TypeError(
        f"You passed a `{old}` object ({type(instance).__qualname__}) to a `{new}`"
        " Trainer. Please switch to a single import style."
    )


_T = TypeVar("_T")  # type of the method owner
_P = ParamSpec("_P")  # parameters of the decorated method
_R_co = TypeVar("_R_co", covariant=True)  # return type of the decorated method


class _restricted_classmethod_impl(Generic[_T, _P, _R_co]):
    """Drop-in replacement for @classmethod, but raises an exception when the decorated method is called on an instance
    instead of a class type."""

    def __init__(self, method: Callable[Concatenate[Type[_T], _P], _R_co]) -> None:
        self.method = method

    def __get__(self, instance: Optional[_T], cls: Type[_T]) -> Callable[_P, _R_co]:
        # The wrapper ensures that the method can be inspected, but not called on an instance
        @functools.wraps(self.method)
        def wrapper(*args: Any, **kwargs: Any) -> _R_co:
            # Workaround for https://github.com/pytorch/pytorch/issues/67146
            is_scripting = any(os.path.join("torch", "jit") in frameinfo.filename for frameinfo in inspect.stack())
            if instance is not None and not is_scripting:
                raise TypeError(
                    f"The classmethod `{cls.__name__}.{self.method.__name__}` cannot be called on an instance."
                    " Please call it on the class type and make sure the return value is used."
                )
            return self.method(cls, *args, **kwargs)

        return wrapper


# trick static type checkers into thinking it's a @classmethod
# https://github.com/microsoft/pyright/issues/5865
_restricted_classmethod = classmethod if TYPE_CHECKING else _restricted_classmethod_impl
