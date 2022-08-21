import json
import queue
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
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


def _push_log_events_to_read_queue_callback(component_name: str, read_queue: queue.PriorityQueue):
    """Pushes _LogEvents from websocket to read_queue.

    Returns callback function used with `on_message_callback` of websocket.WebSocketApp.
    """

    def callback(ws_app: WebSocketApp, msg: str):
        # We strongly trust that the contract on API will hold atm :D
        event_dict = json.loads(msg)
        print(event_dict

              )
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


def _cluster_logs_reader(
    client: LightningClient,
    cluster_id: str,
    follow: bool,
    on_error_callback: Optional[Callable] = None,
) -> Iterator[_LogEvent]:

    logs_api_client = _LightningLogsSocketAPI(client.api_client)
    read_queue = queue.PriorityQueue()

    # We will use a socket per component
    log_socket = logs_api_client.create_cluster_logs_socket(
            cluster_id=cluster_id,
            on_message_callback=_push_log_events_to_read_queue_callback(cluster_id, read_queue),
            on_error_callback=on_error_callback or _error_callback,
        )

    # And each socket on separate thread pushing log event to print queue
    #   run_forever() will run until we close() the connection from outside
    log_thread = Thread(target=log_socket.run_forever)

    # Establish connection and begin pushing logs to the print queue
    log_thread.start()



    # Print logs from queue when log event is available
    try:
        while True:
            log_event = read_queue.get(timeout=None if follow else 1.0)
            yield log_event

            #if user_log_start in log_event.message:
            #    start_timestamp = log_event.timestamp + timedelta(seconds=0.5)

            #if start_timestamp and log_event.timestam
            #
            # p > start_timestamp:
            #    yield log_event

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
