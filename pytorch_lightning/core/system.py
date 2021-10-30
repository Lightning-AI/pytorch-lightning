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
"""The LightningSystem - an LightningModule """
import functools
from collections import OrderedDict
from typing import Callable, Dict, IO, Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torchmetrics import Metric

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException


# https://stackoverflow.com/a/63851681/9201239
def _get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


def _wrap_init(f, system=None):
    @functools.wraps(f)
    def wrapper(module, *args, **kwargs):
        nonlocal system
        f(module, *args, **kwargs)
        system._module_map_to_arguments[id(module)] = (args, kwargs)

    return wrapper


def _enable_class(cls, system: "LightningSystem"):
    cls._old_init = cls.__init__
    cls.__init__ = _wrap_init(cls.__init__, system=system)


def _disable_class(cls):
    cls.__init__ = cls._old_init


def load_from_checkpoint(
    module_cls: Type[Module],
    checkpoint_path: Union[str, IO],
    map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
    **kwargs,
) -> Module:
    if issubclass(module_cls, LightningSystem):
        raise MisconfigurationException("This utility should be used to instantiate your model, not a LightningSystem")
    if map_location is not None:
        checkpoint = pl_load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

    module, module_class_name = checkpoint.pop("module_info")
    assert module_cls.__module__ == module
    assert module_cls.__name__ == module_class_name
    checkpoint.pop("module_name")
    module = module_cls(*checkpoint.pop("args"), **checkpoint.pop("kwargs"))
    module.load_state_dict(checkpoint)
    return module


class LightningSystem(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_modules = OrderedDict()
        self._module_map_to_arguments = {}
        self._args = None
        self._kwargs = None
        for subclass in _get_all_subclasses(Module):
            _enable_class(subclass, self)

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if isinstance(value, Parameter):
            raise MisconfigurationException("A `LightningSystem` doesn't support parameters.")
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError("cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                if isinstance(value, Metric):
                    self._metric_modules[name] = value
                elif len(modules) > 0:
                    raise MisconfigurationException(
                        "A `LightningSystem` supports only a single nn.Module expects `torchmetrics.Metric`."
                    )
                modules[name] = value
                if not isinstance(value, Metric):
                    for subclass in _get_all_subclasses(Module):
                        _disable_class(subclass)
                    self._args, self._kwargs = self._module_map_to_arguments[id(value)]
                    self._module_map_to_arguments = None
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        "cannot assign '{}' as child module '{}' "
                        "(torch.nn.Module or None expected)".format(torch.typename(value), name)
                    )
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if isinstance(buffers, Tensor):
                    raise MisconfigurationException("A `LightningSystem` doesn't support parameters.")
                else:
                    object.__setattr__(self, name, value)

    @property
    def has_module(self) -> bool:
        return len(self.__dict__.get("_modules")) > 0

    @property
    def module_name(self) -> Optional[str]:
        if self.has_module:
            modules: OrderedDict = self.__dict__.get("_modules")
            return list(modules.keys())[0]

    @property
    def module_info(self) -> Optional[str]:
        if self.has_module:
            modules: OrderedDict = self.__dict__.get("_modules")
            module = list(modules.values())[0]
            return (module.__module__, module.__class__.__name__)

    def state_dict(self, destination=None, prefix="", keep_vars=False, extras: bool = True):
        state_dict = super().state_dict()
        if self.has_module:
            module_name = self.module_name

            def prune_name(k: str) -> str:
                if k.startswith(module_name):
                    return k.replace(module_name + ".", "")
                return k

            state_dict = {prune_name(k): v for k, v in state_dict.items()}
            if extras:
                state_dict["args"] = self._args
                state_dict["kwargs"] = self._kwargs
                state_dict["module_name"] = self.module_name
                state_dict["module_info"] = self.module_info
            return state_dict
        return state_dict

    def load_state_dict(self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True):
        if not self.has_module:
            raise MisconfigurationException("A `LightningSystem doesn't contain any module.")
        module_name = self.module_name

        def add_name(k: str) -> str:
            if "." not in k:
                return module_name + "." + k
            return k

        state_dict = OrderedDict({add_name(k): v for k, v in state_dict.items()})
        return super().load_state_dict(state_dict, strict=strict)


__all__ = ["LightningSystem", "load_from_checkpoint"]
