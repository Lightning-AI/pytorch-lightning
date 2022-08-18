import json
import queue
import sys
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from threading import Thread
from typing import Callable, Iterator, List, Optional

import dateutil.parser
from websocket import WebSocketApp

from lightning_app.utilities.logs_socket_api import _LightningLogsSocketAPI
from lightning_app.utilities.network import LightningClient


@dataclass
class _LogEventLabels:
    app: str
    container: str
    filename: str
    job: str
    namespace: str
    node_name: str
    pod: str
    stream: Optional[str] = None


@dataclass
class _LogEvent:
    message: str
    timestamp: datetime
    component_name: str
    labels: _LogEventLabels

    def __ge__(self, other: "_LogEvent") -> bool:
        return self.timestamp >= other.timestamp

    def __gt__(self, other: "_LogEvent") -> bool:
        return self.timestamp > other.timestamp


class LogSection:
    """Delimiters used to separate sections of Lightning App logs."""

    USER_RUN_FLOW = "USER_RUN_FLOW"
    USER_RUN_WORK = "USER_RUN_WORK"


def is_log_delimiter_begin(message: str):
    return "<<< BEGIN" in message and "SECTION >>>" in message


def is_log_delimiter_end(message: str):
    return "<<< END" in message and "SECTION >>>" in message


def log_delimiter_begin_for(section: str) -> str:
    return f"<<< BEGIN {section} SECTION >>>"


def log_delimiter_end_for(section: str) -> str:
    return f"<<< END {section} SECTION >>>"


def is_log_delimiter(message: str) -> bool:
    return is_log_delimiter_begin(message) or is_log_delimiter_end(message)


def is_internal_launcher_command(message: str) -> bool:
    return "lightning-cloud-launcher" in message


def is_internal_log(message: str) -> bool:
    return is_internal_launcher_command(message) or is_log_delimiter(message)


def _push_log_events_to_read_queue_callback(component_name: str, read_queue: queue.PriorityQueue):
    """Pushes _LogEvents from websocket to read_queue.

    Returns callback function used with `on_message_callback` of websocket.WebSocketApp.
    """

    def callback(ws_app: WebSocketApp, msg: str):
        # We strongly trust that the contract on API will hold atm :D
        event_dict = json.loads(msg)
        labels = _LogEventLabels(**event_dict["labels"])

        if "message" in event_dict:
            message = event_dict["message"]
            timestamp = dateutil.parser.isoparse(event_dict["timestamp"])
            event = _LogEvent(
                message=message,
                timestamp=timestamp,
                component_name=component_name,
                labels=labels,
            )
            read_queue.put(event)

    return callback


def _error_callback(ws_app: WebSocketApp, error: Exception):
    errors = {
        KeyError: "Malformed log message, missing key",
        JSONDecodeError: "Malformed log message",
        TypeError: "Malformed log format",
        ValueError: "Malformed date format",
    }
    print(f"Error while reading logs ({errors.get(type(error), 'Unknown')})", file=sys.stderr)
    ws_app.close()


def _app_logs_reader(
    client: LightningClient,
    project_id: str,
    app_id: str,
    component_names: List[str],
    follow: bool,
    on_error_callback: Optional[Callable] = None,
) -> Iterator[_LogEvent]:
    read_queue = queue.PriorityQueue()
    logs_api_client = _LightningLogsSocketAPI(client.api_client)

    # We will use a socket per component
    log_sockets = [
        logs_api_client.create_lightning_logs_socket(
            project_id=project_id,
            app_id=app_id,
            component=component_name,
            on_message_callback=_push_log_events_to_read_queue_callback(component_name, read_queue),
            on_error_callback=on_error_callback or _error_callback,
        )
        for component_name in component_names
    ]

    # And each socket on separate thread pushing log event to print queue
    #   run_forever() will run until we close() the connection from outside
    log_threads = [Thread(target=work.run_forever) for work in log_sockets]

    # Establish connection and begin pushing logs to the print queue
    for th in log_threads:
        th.start()

    # Print logs from queue when log event is available
    component_logs_started = {component_name: False for component_name in component_names}

    try:
        while True:
            log_event: _LogEvent = read_queue.get(timeout=None if follow else 1.0)

            if component_logs_started[log_event.component_name] and not is_internal_log(log_event.message):
                yield log_event
                continue

            section = LogSection.USER_RUN_FLOW if log_event.component_name == "flow" else LogSection.USER_RUN_WORK

            if log_delimiter_begin_for(section) in log_event.message:
                component_logs_started[log_event.component_name] = True

    except queue.Empty:
        # Empty is raised by queue.get if timeout is reached. Follow = False case.
        pass

    except KeyboardInterrupt:
        # User pressed CTRL+C to exit, we sould respect that
        pass

    finally:
        # Close connections - it will cause run_forever() to finish -> thread as finishes aswell
        for socket in log_sockets:
            socket.close()

        # Because all socket were closed, we can just wait for threads to finish.
        for th in log_threads:
            th.join()
