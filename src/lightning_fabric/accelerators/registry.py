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
import importlib
from inspect import getmembers, isclass
from typing import Any, Callable, Dict, List, Optional

from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.registry import _is_register_method_overridden


class _AcceleratorRegistry(dict):
    """This class is a Registry that stores information about the Accelerators.

    The Accelerators are mapped to strings. These strings are names that identify
    an accelerator, e.g., "gpu". It also returns Optional description and
    parameters to initialize the Accelerator, which were defined during the
    registration.

    The motivation for having a AcceleratorRegistry is to make it convenient
    for the Users to try different accelerators by passing mapped aliases
    to the accelerator flag to the Trainer.

    Example::

        @AcceleratorRegistry.register("sota", description="Custom sota accelerator", a=1, b=True)
        class SOTAAccelerator(Accelerator):
            def __init__(self, a, b):
                ...

        or

        AcceleratorRegistry.register("sota", SOTAAccelerator, description="Custom sota accelerator", a=1, b=True)
    """

    def register(
        self,
        name: str,
        accelerator: Optional[Callable] = None,
        description: str = "",
        override: bool = False,
        **init_params: Any,
    ) -> Callable:
        """Registers a accelerator mapped to a name and with required metadata.

        Args:
            name : the name that identifies a accelerator, e.g. "gpu"
            accelerator : accelerator class
            description : accelerator description
            override : overrides the registered accelerator, if True
            init_params: parameters to initialize the accelerator
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"`name` must be a str, found {name}")

        if name in self and not override:
            raise MisconfigurationException(f"'{name}' is already present in the registry. HINT: Use `override=True`.")

        data: Dict[str, Any] = {}

        data["description"] = description
        data["init_params"] = init_params

        def do_register(name: str, accelerator: Callable) -> Callable:
            data["accelerator"] = accelerator
            data["accelerator_name"] = name
            self[name] = data
            return accelerator

        if accelerator is not None:
            return do_register(name, accelerator)

        return do_register

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """Calls the registered accelerator with the required parameters and returns the accelerator object.

        Args:
            name (str): the name that identifies a accelerator, e.g. "gpu"
        """
        if name in self:
            data = self[name]
            return data["accelerator"](**data["init_params"])

        if default is not None:
            return default

        err_msg = "'{}' not found in registry. Available names: {}"
        available_names = self.available_accelerators()
        raise KeyError(err_msg.format(name, available_names))

    def remove(self, name: str) -> None:
        """Removes the registered accelerator by name."""
        self.pop(name)

    def available_accelerators(self) -> List[str]:
        """Returns a list of registered accelerators."""
        return list(self.keys())

    def __str__(self) -> str:
        return "Registered Accelerators: {}".format(", ".join(self.available_accelerators()))


def call_register_accelerators(registry: _AcceleratorRegistry, base_module: str) -> None:
    module = importlib.import_module(base_module)
    for _, mod in getmembers(module, isclass):
        if issubclass(mod, Accelerator) and _is_register_method_overridden(mod, Accelerator, "register_accelerators"):
            mod.register_accelerators(registry)
