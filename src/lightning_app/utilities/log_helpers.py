import sys
from json import JSONDecodeError
from dataclasses import dataclass
from datetime import datetime

from websocket import WebSocketApp


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
    print(f"Error while reading logs ({errors.get(type(error), 'Unknown')}), {error}", file=sys.stderr)
    ws_app.close()
