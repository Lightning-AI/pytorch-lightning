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
import inspect
from collections import UserDict
from inspect import getmembers, isclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class _TrainingTypePluginsRegistry(UserDict):
    """
    This class is a Registry that stores information about the Training Type Plugins.

    The Plugins are mapped to strings. These strings are names that idenitify
    a plugin, e.g., "deepspeed". It also returns Optional description and
    parameters to initialize the Plugin, which were defined durng the
    registration.

    The motivation for having a TrainingTypePluginRegistry is to make it convenient
    for the Users to try different Plugins by passing just strings
    to the plugins flag to the Trainer.

    Example::

        @TrainingTypePluginsRegistry.register("lightning", description="Super fast", a=1, b=True)
        class LightningPlugin:
            def __init__(self, a, b):
                ...

        or

        TrainingTypePluginsRegistry.register("lightning", LightningPlugin, description="Super fast", a=1, b=True)

    """

    def register(
        self,
        name: str,
        plugin: Optional[Callable] = None,
        description: Optional[str] = None,
        override: bool = False,
        **init_params: Any,
    ) -> Callable:
        """
        Registers a plugin mapped to a name and with required metadata.

        Args:
            name : the name that identifies a plugin, e.g. "deepspeed_stage_3"
            plugin : plugin class
            description : plugin description
            override : overrides the registered plugin, if True
            init_params: parameters to initialize the plugin
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'`name` must be a str, found {name}')

        if name in self and not override:
            raise MisconfigurationException(
                f"'{name}' is already present in the registry."
                " HINT: Use `override=True`."
            )

        data: Dict[str, Any] = {}
        data["description"] = description if description is not None else ""

        data["init_params"] = init_params

        def do_register(plugin: Callable) -> Callable:
            data["plugin"] = plugin
            self[name] = data
            return plugin

        if plugin is not None:
            return do_register(plugin)

        return do_register

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """
        Calls the registered plugin with the required parameters
        and returns the plugin object

        Args:
            name (str): the name that identifies a plugin, e.g. "deepspeed_stage_3"
        """
        if name in self:
            data = self[name]
            return data["plugin"](**data["init_params"])

        if default is not None:
            return default

        err_msg = "'{}' not found in registry. Available names: {}"
        available_names = ", ".join(sorted(self.keys())) or "none"
        raise KeyError(err_msg.format(name, available_names))

    def remove(self, name: str) -> None:
        """Removes the registered plugin by name"""
        self.pop(name)

    def available_plugins(self) -> List:
        """Returns a list of registered plugins"""
        return list(self.keys())

    def __str__(self) -> str:
        return "Registered Plugins: {}".format(", ".join(self.keys()))


TrainingTypePluginsRegistry = _TrainingTypePluginsRegistry()


def is_register_plugins_overridden(plugin: type) -> bool:

    method_name = "register_plugins"
    plugin_attr = getattr(plugin, method_name)
    previous_super_cls = inspect.getmro(plugin)[1]

    if issubclass(previous_super_cls, TrainingTypePlugin):
        super_attr = getattr(previous_super_cls, method_name)
    else:
        return False

    if hasattr(plugin_attr, 'patch_loader_code'):
        is_overridden = plugin_attr.patch_loader_code != str(super_attr.__code__)
    else:
        is_overridden = plugin_attr.__code__ is not super_attr.__code__
    return is_overridden


def call_training_type_register_plugins(root: Path, base_module: str) -> None:
    module = importlib.import_module(base_module)
    for _, mod in getmembers(module, isclass):
        if issubclass(mod, TrainingTypePlugin) and is_register_plugins_overridden(mod):
            mod.register_plugins(TrainingTypePluginsRegistry)
