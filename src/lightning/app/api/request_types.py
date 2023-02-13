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

from dataclasses import asdict, dataclass
from typing import Any, Optional

from deepdiff import Delta


@dataclass
class _BaseRequest:
    def to_dict(self):
        return asdict(self)


@dataclass
class _DeltaRequest(_BaseRequest):
    delta: Delta

    def to_dict(self):
        return self.delta.to_dict()


@dataclass
class _CommandRequest(_BaseRequest):
    id: str
    name: str
    method_name: str
    args: Any
    kwargs: Any


@dataclass
class _APIRequest(_BaseRequest):
    id: str
    name: str
    method_name: str
    args: Any
    kwargs: Any


@dataclass
class _RequestResponse(_BaseRequest):
    status_code: int
    content: Optional[str] = None
