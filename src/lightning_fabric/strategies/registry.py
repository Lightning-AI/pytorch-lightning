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

from lightning_fabric.strategies.strategy import Strategy
from lightning_fabric.utilities.registry import _is_register_method_overridden


class _StrategyRegistry(dict):
    """This class is a Registry that stores information about the Training Strategies.

    The Strategies are mapped to strings. These strings are names that identify
    a strategy, e.g., "deepspeed". It also returns Optional description and
    parameters to initialize the Strategy, which were defined durng the
    registration.

    The motivation for having a StrategyRegistry is to make it convenient
    for the Users to try different Strategies by passing just strings
    to the strategy flag to the Trainer.

    Example::

        @StrategyRegistry.register("lightning", description="Super fast", a=1, b=True)
        class LightningStrategy:
            def __init__(self, a, b):
                ...

        or

        StrategyRegistry.register("lightning", LightningStrategy, description="Super fast", a=1, b=True)
    """

    def register(
        self,
        name: str,
        strategy: Optional[Callable] = None,
        description: Optional[str] = None,
        override: bool = False,
        **init_params: Any,
    ) -> Callable:
        """Registers a strategy mapped to a name and with required metadata.

        Args:
            name : the name that identifies a strategy, e.g. "deepspeed_stage_3"
            strategy : strategy class
            description : strategy description
            override : overrides the registered strategy, if True
            init_params: parameters to initialize the strategy
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"`name` must be a str, found {name}")

        if name in self and not override:
            raise ValueError(f"'{name}' is already present in the registry. HINT: Use `override=True`.")

        data: Dict[str, Any] = {}
        data["description"] = description if description is not None else ""

        data["init_params"] = init_params

        def do_register(strategy: Callable) -> Callable:
            data["strategy"] = strategy
            data["strategy_name"] = name
            self[name] = data
            return strategy

        if strategy is not None:
            return do_register(strategy)

        return do_register

    def get(self, name: str, default: Optional[Strategy] = None) -> Strategy:  # type: ignore[override]
        """Calls the registered strategy with the required parameters and returns the strategy object.

        Args:
            name (str): the name that identifies a strategy, e.g. "deepspeed_stage_3"
        """
        if name in self:
            data = self[name]
            return data["strategy"](**data["init_params"])

        if default is not None:
            return default

        err_msg = "'{}' not found in registry. Available names: {}"
        available_names = ", ".join(sorted(self.keys())) or "none"
        raise KeyError(err_msg.format(name, available_names))

    def remove(self, name: str) -> None:
        """Removes the registered strategy by name."""
        self.pop(name)

    def available_strategies(self) -> List:
        """Returns a list of registered strategies."""
        return list(self.keys())

    def __str__(self) -> str:
        return "Registered Strategies: {}".format(", ".join(self.keys()))


def _call_register_strategies(registry: _StrategyRegistry, base_module: str) -> None:
    module = importlib.import_module(base_module)
    for _, mod in getmembers(module, isclass):
        if issubclass(mod, Strategy) and _is_register_method_overridden(mod, Strategy, "register_strategies"):
            mod.register_strategies(registry)
