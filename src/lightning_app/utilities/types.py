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

import typing as t

from typing_extensions import Protocol, runtime_checkable

from lightning_app import LightningFlow, LightningWork
from lightning_app.structures import Dict, List

Component = t.Union[LightningFlow, LightningWork, Dict, List]
ComponentTuple = (LightningFlow, LightningWork, Dict, List)


@runtime_checkable
class Hashable(Protocol):
    def to_dict(self) -> t.Dict[str, t.Any]:
        ...
