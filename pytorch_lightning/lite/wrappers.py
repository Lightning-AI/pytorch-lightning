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
import inspect
from contextlib import contextmanager
from itertools import chain
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, Optional, Set, Sized, Type, Union

import torch
from torch import nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin
from pytorch_lightning.plugins import PrecisionPlugin
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
        # `__del__` is skipped in case the optimizer has implemented custom destructor logic which we would
        # not want to call on destruction of the `_LiteOptimizer
        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k not in ("step", "__del__")}
        self.__class__ = type("Lite" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})
        self._optimizer = optimizer
        self._accelerator = accelerator

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def step(self, closure: Optional[Callable] = None) -> None:
        closure = closure or _do_nothing_closure
        self._accelerator.optimizer_step(
            self.optimizer,
            opt_idx=0,
            closure=closure,
            model=self._accelerator.model,
        )


class _LiteModule(DeviceDtypeModuleMixin):
    def __init__(self, module: nn.Module, precision_plugin: PrecisionPlugin) -> None:
        """The LiteModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            module: The module to wrap
            precision_plugin: Reference to the precision plugin for handling precision context
        """
        super().__init__()
        self._module = module
        self._precision_plugin = precision_plugin

    @property
    def module(self) -> nn.Module:
        return self._module

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
            output = self.module(*args, **kwargs)

        to_type = torch.get_default_dtype()
        output = apply_to_collection(output, function=_convert_float_tensor, dtype=Tensor)
        return output


def _wrap_init(f: Callable) -> Callable:
    @functools.wraps(f)
    def wrapper(module: Any, *args: Any, **kwargs: Dict[str, Any]) -> None:
        params = dict(inspect.signature(module._old_init).parameters)
        params.pop("args", None)
        params.pop("kwargs", None)
        for init_name, init_arg in chain(zip(params, args), kwargs.items()):
            setattr(module, init_name, init_arg)
        f(module, *args, **kwargs)

    return wrapper


# https://stackoverflow.com/a/63851681/9201239
def _get_all_subclasses(cls: Type[Any]) -> Set[Type[Any]]:
    subclasses = set()

    def recurse(cl: Type[Any]) -> None:
        for subclass in cl.__subclasses__():
            subclasses.add(subclass)
            recurse(subclass)

    recurse(cls)
    return subclasses


def _enable_class(cls: Type[Any]) -> None:
    cls._old_init = cls.__init__
    cls.__init__ = _wrap_init(cls.__init__)


def _disable_class(cls: Type[Any]) -> None:
    cls.__init__ = cls._old_init
    del cls._old_init


@contextmanager
def _replace_dataloader_init_method() -> Generator:
    """This context manager is used to support custom :class:`~torch.utils.data.DataLoader."""
    for subclass in _get_all_subclasses(DataLoader):
        _enable_class(subclass)
    yield
    for subclass in _get_all_subclasses(DataLoader):
        _disable_class(subclass)


class _LiteDataLoader:
    def __init__(self, dataloader: Union[Iterable, DataLoader], device: Optional[torch.device] = None) -> None:
        """The LiteDataLoader is an extension of an Iterator. It would move the data to the device automatically if
        the device is specified.

        Args:
            dataloader: The current dataloader to be used.
            device: The device to which the data should be moved. By default the device is `None` and no data
                transfers will be made (identical behavior as :class:`~torch.utils.data.DataLoader`).
        """
        super().__init__()
        self.__dict__.update(getattr(dataloader, "__dict__", {}))
        self._dataloader = dataloader
        self._device = device

    def __len__(self) -> Union[int, float]:
        if isinstance(self._dataloader, Sized):
            return len(self._dataloader)
        return float("inf")

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def __iter__(self) -> Union[Iterator[Any], Generator[Any, None, None]]:
        iterator = iter(self._dataloader)
        if self._device is None:
            yield from iterator

        for item in iterator:
            yield move_data_to_device(item, self._device)
