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

from enum import Enum
from typing import Type, TYPE_CHECKING

from lightning_app.runners import CloudRuntime, MultiProcessRuntime

if TYPE_CHECKING:
    from lightning_app.runners.runtime import Runtime


class RuntimeType(Enum):
    MULTIPROCESS = "multiprocess"
    CLOUD = "cloud"

    def get_runtime(self) -> Type["Runtime"]:
        if self == RuntimeType.MULTIPROCESS:
            return MultiProcessRuntime
        elif self == RuntimeType.CLOUD:
            return CloudRuntime
        else:
            raise ValueError("Unknown runtime type")
