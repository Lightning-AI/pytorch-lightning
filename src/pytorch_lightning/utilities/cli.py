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
"""Deprecated utilities for LightningCLI."""

import inspect
from types import ModuleType
from typing import Any, Generator, List, Optional, Tuple, Type

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
import pytorch_lightning.cli as new_cli
from pytorch_lightning.utilities.meta import get_all_subclasses
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation

_deprecate_registry_message = (
    "`LightningCLI`'s registries were deprecated in v1.7 and will be removed "
    "in v1.9. Now any imported subclass is automatically available by name in "
    "`LightningCLI` without any need to explicitly register it."
)

_deprecate_auto_registry_message = (
    "`LightningCLI.auto_registry` parameter was deprecated in v1.7 and will be removed "
    "in v1.9. Now any imported subclass is automatically available by name in "
    "`LightningCLI` without any need to explicitly register it."
)


class _Registry(dict):  # Remove in v1.9
    def __call__(
        self, cls: Type, key: Optional[str] = None, override: bool = False, show_deprecation: bool = True
    ) -> Type:
        """Registers a class mapped to a name.

        Args:
            cls: the class to be mapped.
            key: the name that identifies the provided class.
            override: Whether to override an existing key.
        """
        if key is None:
            key = cls.__name__
        elif not isinstance(key, str):
            raise TypeError(f"`key` must be a str, found {key}")

        if key not in self or override:
            self[key] = cls

        self._deprecation(show_deprecation)
        return cls

    def register_classes(
        self, module: ModuleType, base_cls: Type, override: bool = False, show_deprecation: bool = True
    ) -> None:
        """This function is an utility to register all classes from a module."""
        for cls in self.get_members(module, base_cls):
            self(cls=cls, override=override, show_deprecation=show_deprecation)

    @staticmethod
    def get_members(module: ModuleType, base_cls: Type) -> Generator[Type, None, None]:
        return (
            cls
            for _, cls in inspect.getmembers(module, predicate=inspect.isclass)
            if issubclass(cls, base_cls) and cls != base_cls
        )

    @property
    def names(self) -> List[str]:
        """Returns the registered names."""
        self._deprecation()
        return list(self.keys())

    @property
    def classes(self) -> Tuple[Type, ...]:
        """Returns the registered classes."""
        self._deprecation()
        return tuple(self.values())

    def __str__(self) -> str:
        return f"Registered objects: {self.names}"

    def _deprecation(self, show_deprecation: bool = True) -> None:
        if show_deprecation and not getattr(self, "deprecation_shown", False):
            rank_zero_deprecation(_deprecate_registry_message)
            self.deprecation_shown = True


OPTIMIZER_REGISTRY = _Registry()
LR_SCHEDULER_REGISTRY = _Registry()
CALLBACK_REGISTRY = _Registry()
MODEL_REGISTRY = _Registry()
DATAMODULE_REGISTRY = _Registry()
LOGGER_REGISTRY = _Registry()


def _populate_registries(subclasses: bool) -> None:  # Remove in v1.9
    if subclasses:
        rank_zero_deprecation(_deprecate_auto_registry_message)
        # this will register any subclasses from all loaded modules including userland
        for cls in get_all_subclasses(torch.optim.Optimizer):
            OPTIMIZER_REGISTRY(cls, show_deprecation=False)
        for cls in get_all_subclasses(torch.optim.lr_scheduler._LRScheduler):
            LR_SCHEDULER_REGISTRY(cls, show_deprecation=False)
        for cls in get_all_subclasses(pl.Callback):
            CALLBACK_REGISTRY(cls, show_deprecation=False)
        for cls in get_all_subclasses(pl.LightningModule):
            MODEL_REGISTRY(cls, show_deprecation=False)
        for cls in get_all_subclasses(pl.LightningDataModule):
            DATAMODULE_REGISTRY(cls, show_deprecation=False)
        for cls in get_all_subclasses(pl.loggers.Logger):
            LOGGER_REGISTRY(cls, show_deprecation=False)
    else:
        # manually register torch's subclasses and our subclasses
        OPTIMIZER_REGISTRY.register_classes(torch.optim, Optimizer, show_deprecation=False)
        LR_SCHEDULER_REGISTRY.register_classes(
            torch.optim.lr_scheduler, torch.optim.lr_scheduler._LRScheduler, show_deprecation=False
        )
        CALLBACK_REGISTRY.register_classes(pl.callbacks, pl.Callback, show_deprecation=False)
        LOGGER_REGISTRY.register_classes(pl.loggers, pl.loggers.Logger, show_deprecation=False)
    # `ReduceLROnPlateau` does not subclass `_LRScheduler`
    LR_SCHEDULER_REGISTRY(cls=new_cli.ReduceLROnPlateau, show_deprecation=False)


def _deprecation(cls: Type) -> None:
    rank_zero_deprecation(
        f"`pytorch_lightning.utilities.cli.{cls.__name__}` has been deprecated in v1.7 and will be removed in v1.9."
        f" Use the equivalent class in `pytorch_lightning.cli.{cls.__name__}` instead."
    )


class LightningArgumentParser(new_cli.LightningArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _deprecation(type(self))
        super().__init__(*args, **kwargs)


class SaveConfigCallback(new_cli.SaveConfigCallback):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _deprecation(type(self))
        super().__init__(*args, **kwargs)


class LightningCLI(new_cli.LightningCLI):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _deprecation(type(self))
        super().__init__(*args, **kwargs)


def instantiate_class(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.cli.instantiate_class` has been deprecated in v1.7 and will be removed in v1.9."
        " Use the equivalent function in `pytorch_lightning.cli.instantiate_class` instead."
    )
    return new_cli.instantiate_class(*args, **kwargs)
