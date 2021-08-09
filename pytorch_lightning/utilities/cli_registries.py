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
import inspect
from collections import UserDict
from typing import Any, Callable, List, Optional, Type

import torch

import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class Registry(UserDict):
    def __call__(
        self,
        cls: Optional[Type] = None,
        key: Optional[str] = None,
        override: bool = False,
    ) -> Callable:
        """
        Registers a plugin mapped to a name and with required metadata.

        Args:
            key : the name that identifies a plugin, e.g. "deepspeed_stage_3"
            value : plugin class
        """
        if key is None:
            key = cls.__name__
        elif not isinstance(key, str):
            raise TypeError(f"`key` must be a str, found {key}")

        if key in self and not override:
            raise MisconfigurationException(f"'{key}' is already present in the registry. HINT: Use `override=True`.")

        def do_register(key, cls) -> Callable:
            self[key] = cls
            return cls

        do_register(key, cls)

        return do_register

    def register_package(self, module, base_cls: Type) -> None:
        for obj_name in dir(module):
            obj_cls = getattr(module, obj_name)
            if inspect.isclass(obj_cls) and issubclass(obj_cls, base_cls):
                self(cls=obj_cls)

    def get(self, name: Optional[str], default: Optional[Any] = None) -> Any:
        """
        Calls the registered plugin with the required parameters
        and returns the plugin object

        Args:
            name (str): the name
        """
        if name in self:
            return self[name]
        else:
            raise KeyError

    def remove(self, name: str) -> None:
        """Removes the registered plugin by name"""
        self.pop(name)

    def available_objects(self) -> List:
        """Returns a list of registered plugins"""
        return list(self.keys())

    def __str__(self) -> str:
        return "Registered Plugins: {}".format(", ".join(self.keys()))


CALLBACK_REGISTRIES = Registry()
CALLBACK_REGISTRIES.register_package(pl.callbacks, pl.callbacks.Callback)

OPTIMIZER_REGISTRIES = Registry()
OPTIMIZER_REGISTRIES.register_package(torch.optim, torch.optim.Optimizer)

SCHEDULER_REGISTRIES = Registry()
SCHEDULER_REGISTRIES.register_package(torch.optim.lr_scheduler, torch.optim.lr_scheduler._LRScheduler)
