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
import importlib
from inspect import getmembers, isclass
from pathlib import Path
from typing import Any, List, Optional

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class _AcceleratorRegistry(dict):
    def register(self, accelerator: Accelerator, name: Optional[str] = None, override: bool = False) -> Accelerator:
        """Registers an accelerator mapped to a name.

        Args:
            accelerator: the accelerator to be mapped.
            name: the name that identifies the provided accelerator.
            override: Whether to override an existing key.
        """
        if name is None:
            name = accelerator.name()
        elif not isinstance(name, str):
            raise TypeError(f"`name` must be a str, found {name}")

        if name in self and not override:
            raise MisconfigurationException(f"'{name}' is already present in the registry. HINT: Use `override=True`.")
        self[name] = accelerator
        return accelerator

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """Calls the registered Accelerator and returns the Accelerator object.

        Args:
            name (str): the name that identifies a Accelerator, e.g. "tpu"
        """
        if name in self:
            accelerator = self[name]
            return accelerator()

        if default is not None:
            return default

        err_msg = "'{}' not found in registry. Available names: {}"
        available_names = ", ".join(sorted(self.keys())) or "none"
        raise KeyError(err_msg.format(name, available_names))

    def remove(self, name: str) -> None:
        """Removes the registered accelerator by name."""
        self.pop(name)

    def available_accelerators(self) -> List:
        """Returns a list of registered accelerators."""
        return list(self.keys())

    def __str__(self) -> str:
        return "Registered Accelerators: {}".format(", ".join(self.keys()))


AcceleratorRegistry = _AcceleratorRegistry()


def register_accelerators(root: Path, base_module: str) -> None:
    module = importlib.import_module(base_module)
    for _, mod in getmembers(module, isclass):
        if issubclass(mod, Accelerator) and mod is not Accelerator:
            AcceleratorRegistry.register(mod)
