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
import logging
from importlib.metadata import entry_points
from inspect import getmembers, isclass
from types import ModuleType
from typing import Any, Union

from lightning_utilities import is_overridden

from lightning.fabric.utilities.imports import _PYTHON_GREATER_EQUAL_3_10_0

_log = logging.getLogger(__name__)


def _load_external_callbacks(group: str) -> list[Any]:
    """Collect external callbacks registered through entry points.

    The entry points are expected to be functions returning a list of callbacks.

    Args:
        group: The entry point group name to load callbacks from.

    Return:
        A list of all callbacks collected from external factories.

    """
    factories = (
        entry_points(group=group) if _PYTHON_GREATER_EQUAL_3_10_0 else entry_points().get(group, {})  # type: ignore[arg-type]
    )

    external_callbacks: list[Any] = []
    for factory in factories:
        callback_factory = factory.load()
        callbacks_list: Union[list[Any], Any] = callback_factory()
        callbacks_list = [callbacks_list] if not isinstance(callbacks_list, list) else callbacks_list
        if callbacks_list:
            _log.info(
                f"Adding {len(callbacks_list)} callbacks from entry point '{factory.name}':"
                f" {', '.join(type(cb).__name__ for cb in callbacks_list)}"
            )
        external_callbacks.extend(callbacks_list)
    return external_callbacks


def _register_classes(registry: Any, method: str, module: ModuleType, parent: type[object]) -> None:
    for _, member in getmembers(module, isclass):
        if issubclass(member, parent) and is_overridden(method, member, parent):
            register_fn = getattr(member, method)
            register_fn(registry)
