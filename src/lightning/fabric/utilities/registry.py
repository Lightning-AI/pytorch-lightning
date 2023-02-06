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
from typing import Any


def _is_register_method_overridden(mod: type, base_cls: Any, method: str) -> bool:
    mod_attr = getattr(mod, method)
    previous_super_cls = inspect.getmro(mod)[1]

    if issubclass(previous_super_cls, base_cls):
        super_attr = getattr(previous_super_cls, method)
    else:
        return False

    return mod_attr.__code__ is not super_attr.__code__
