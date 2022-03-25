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
from typing import Any, List, Optional, Type

from pytorch_lightning.accelerators.accelerator import Accelerator


class _AcceleratorRegistry(dict):
    """This class is a dictionary that stores information about the Accelerators.

    The Accelerators are mapped to strings. These strings are names that identify
    an accelerator, e.g., "gpu". It also includes an optional description and any
    parameters to initialize the Accelerator, which were defined during the
    registration.

    The motivation for having a AcceleratorRegistry is to make it convenient
    for the Users to try different accelerators by passing mapped aliases
    to the accelerator flag to the Trainer.

    Example::

        @ACCELERATOR_REGISTRY
        class SOTAAccelerator(Accelerator):
            def __init__(self, a):
                ...

            @staticmethod
            def name():
                return "sota"

        # or to pass parameters
        ACCELERATOR_REGISTRY.register(SOTAAccelerator, description="My SoTA accelerator", a=1)
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Type:
        return self.register(*args, **kwargs)

    def register(
        self,
        accelerator: Type[Accelerator],
        name: Optional[str] = None,
        description: Optional[str] = None,
        override: bool = False,
        **kwargs: Any,
    ) -> Type:
        """Registers an accelerator mapped to a name and with optional metadata.

        Args:
            accelerator: The accelerator class.
            name: The alias for the accelerator, e.g. ``"gpu"``.
            description: An optional description.
            override: Whether to override the registered accelerator.
            **kwargs: parameters to initialize the accelerator.
        """
        if name is None:
            name = accelerator.name()
        if not isinstance(name, str):
            raise TypeError(f"`name` for {accelerator} must be a str, found {name!r}")

        if name not in self or override:
            self[name] = {
                "accelerator": accelerator,
                "description": description if description is not None else accelerator.__class__.__name__,
                "kwargs": kwargs,
            }
        return accelerator

    def get(self, name: str, default: Optional[Accelerator] = None) -> Accelerator:
        """Calls the registered accelerator with the required parameters and returns the accelerator object.

        Args:
            name: The name that identifies a accelerator, e.g. "gpu".
            default: A default value.

        Raises:
            KeyError: If the key does not exist.
        """
        if name in self:
            data = self[name]
            return data["accelerator"](**data["kwargs"])
        if default is not None:
            return default
        raise KeyError(f"{name!r} not found in registry. {self!s}")

    @property
    def names(self) -> List[str]:
        """Returns the registered names."""
        return sorted(list(self))

    def __str__(self) -> str:
        return f"Registered Accelerators: {self.names}"


ACCELERATOR_REGISTRY = _AcceleratorRegistry()
