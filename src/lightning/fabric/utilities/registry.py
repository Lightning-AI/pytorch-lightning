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
import inspect
import logging
from typing import Any, List, Union

from lightning.fabric.utilities.imports import _PYTHON_GREATER_EQUAL_3_8_0, _PYTHON_GREATER_EQUAL_3_10_0

_log = logging.getLogger(__name__)


def _is_register_method_overridden(mod: type, base_cls: Any, method: str) -> bool:
    mod_attr = getattr(mod, method)
    previous_super_cls = inspect.getmro(mod)[1]

    if issubclass(previous_super_cls, base_cls):
        super_attr = getattr(previous_super_cls, method)
    else:
        return False

    return mod_attr.__code__ is not super_attr.__code__


def _load_external_callbacks(group: str) -> List[Any]:
    """Collect external callbacks registered through entry points.

    The entry points are expected to be functions returning a list of callbacks.

    Args:
        group: The entry point group name to load callbacks from.

    Return:
        A list of all callbacks collected from external factories.

    """
    if _PYTHON_GREATER_EQUAL_3_8_0:
        from importlib.metadata import entry_points

        factories = (
            entry_points(group=group)
            if _PYTHON_GREATER_EQUAL_3_10_0
            else entry_points().get(group, {})  # type: ignore[arg-type]
        )
    else:
        from pkg_resources import iter_entry_points

        factories = iter_entry_points(group)  # type: ignore[assignment]

    external_callbacks: List[Any] = []
    for factory in factories:
        callback_factory = factory.load()
        callbacks_list: Union[List[Any], Any] = callback_factory()
        callbacks_list = [callbacks_list] if not isinstance(callbacks_list, list) else callbacks_list
        if callbacks_list:
            _log.info(
                f"Adding {len(callbacks_list)} callbacks from entry point '{factory.name}':"
                f" {', '.join(type(cb).__name__ for cb in callbacks_list)}"
            )
        external_callbacks.extend(callbacks_list)
    return external_callbacks
