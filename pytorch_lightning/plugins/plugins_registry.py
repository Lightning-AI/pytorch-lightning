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
from collections import UserDict
from typing import Any, Callable, List, Optional


class _PluginsRegistry(UserDict):
    """
    This class is a Registry that stores information about the Plugins.

    The Plugins are mapped to strings. These strings are names that idenitify
    a plugin, eg., "deepspeed". It also returns Optional description and
    parameters to initialize the Plugin, which were defined durng the
    registeration.

    The motivation for having a PluginRegistry is to make it convenient
    for the Users to try different Plugins by passing just strings
    to the plugins flag to the Trainer.

    """

    def register(self, name: str, func: Optional[Callable] = None, description: Optional[str] = None, **init_params):

        if not (name is None or isinstance(name, str)):
            raise TypeError(f'`name` must be a str, found {name}')

        data = {}
        data["description"] = description if description is not None else ""

        data["init_params"] = init_params

        def do_register(func):
            data["func"] = func
            self[name] = data
            return data

        if func is not None:
            return do_register(func)

        return do_register

    def get(self, name: str):
        if name in self:
            return self[name]
        raise KeyError("Key not Found")

    def remove(self, name: str):
        self.pop(name)

    def available_plugins(self) -> List:
        return list(self.keys())

    def __str__(self):
        return "Registered Plugins: {}".format(", ".join(self.keys()))


PluginsRegistry = _PluginsRegistry()
