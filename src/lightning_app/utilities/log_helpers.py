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
from datetime import datetime
from json import JSONDecodeError

from websocket import WebSocketApp

from lightning_app.utilities.app_helpers import Logger

logger = Logger(__name__)


# This is a superclass to inherit log entry classes from it:
# it implements magic methods to sort logs by timestamps.
@dataclass
class _OrderedLogEntry:
    message: str
    timestamp: datetime

    def __ge__(self, other: "_OrderedLogEntry") -> bool:
        return self.timestamp >= other.timestamp

    def __gt__(self, other: "_OrderedLogEntry") -> bool:
        return self.timestamp > other.timestamp


# A general error callback for log reading, prints most common types of possible errors.
def _error_callback(ws_app: WebSocketApp, error: Exception):
    errors = {
        KeyError: "Malformed log message, missing key",
        JSONDecodeError: "Malformed log message",
        TypeError: "Malformed log format",
        ValueError: "Malformed date format",
    }
    logger.error(f"⚡ Error while reading logs ({errors.get(type(error), 'Unknown')}), {error}")
    ws_app.close()
