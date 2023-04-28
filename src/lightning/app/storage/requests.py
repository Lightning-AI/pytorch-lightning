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

from dataclasses import dataclass
from typing import Optional


@dataclass
class _GetRequest:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""


@dataclass
class _GetResponse:
    source: str
    name: str
    path: str
    hash: str
    size: int = 0
    destination: str = ""
    exception: Optional[Exception] = None
    timedelta: Optional[float] = None


@dataclass
class _ExistsRequest:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""


@dataclass
class _ExistsResponse:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""
    exists: Optional[bool] = None
    timedelta: Optional[float] = None
