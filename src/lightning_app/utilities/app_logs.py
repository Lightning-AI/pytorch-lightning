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

import json
import queue
from dataclasses import dataclass
from threading import Thread
from typing import Callable, Iterator, List, Optional

import dateutil.parser
from websocket import WebSocketApp

from lightning_app.utilities.log_helpers import _error_callback, _OrderedLogEntry
from lightning_app.utilities.logs_socket_api import _LightningLogsSocketAPI


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
class _LogEvent(_OrderedLogEntry):
    component_name: str
    labels: _LogEventLabels


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


def _app_logs_reader(
    logs_api_client: _LightningLogsSocketAPI,
    project_id: str,
    app_id: str,
    component_names: List[str],
    follow: bool,
    on_error_callback: Optional[Callable] = None,
) -> Iterator[_LogEvent]:

    read_queue = queue.PriorityQueue()

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
    log_threads = [Thread(target=work.run_forever, daemon=True) for work in log_sockets]

    # Establish connection and begin pushing logs to the print queue
    for th in log_threads:
        th.start()

    # Print logs from queue when log event is available
    flow = "Your app has started."
    work = "USER_RUN_WORK"
    start_timestamps = {}

    # Print logs from queue when log event is available
    try:
        while True:
            log_event: _LogEvent = read_queue.get(timeout=None if follow else 1.0)

            token = flow if log_event.component_name == "flow" else work
            if token in log_event.message:
                start_timestamps[log_event.component_name] = log_event.timestamp

            timestamp = start_timestamps.get(log_event.component_name, None)
            if timestamp and log_event.timestamp >= timestamp:
                if "launcher" not in log_event.message:
                    yield log_event

    except queue.Empty:
        # Empty is raised by queue.get if timeout is reached. Follow = False case.
        pass

    except KeyboardInterrupt:
        # User pressed CTRL+C to exit, we should respect that
        pass

    finally:
        # Close connections - it will cause run_forever() to finish -> thread as finishes aswell
        for socket in log_sockets:
            socket.close()

        # Because all socket were closed, we can just wait for threads to finish.
        for th in log_threads:
            th.join()
