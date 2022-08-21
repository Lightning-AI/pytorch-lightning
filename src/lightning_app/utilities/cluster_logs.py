import json
import queue
from dataclasses import dataclass
from threading import Thread
from typing import Callable, Iterator, Optional

import arrow
import dateutil.parser
from websocket import WebSocketApp

from lightning_app.utilities.log_helpers import _error_callback, _OrderedLogEntry
from lightning_app.utilities.logs_socket_api import _ClusterLogsSocketAPI
from lightning_app.utilities.network import LightningClient


@dataclass
class _ClusterLogEventLabels:
    cluster_id: str
    grid_url: str
    hostname: str
    level: str
    logger: str
    path: Optional[str] = None
    workspace: Optional[str] = None
    identifier: Optional[str] = None
    issuer: Optional[str] = None
    error: Optional[str] = None
    errorVerbose: Optional[str] = None


@dataclass
class _ClusterLogEvent(_OrderedLogEntry):
    labels: _ClusterLogEventLabels


def _push_log_events_to_read_queue_callback(read_queue: queue.PriorityQueue):
    """Pushes _LogEvents from websocket to read_queue.

    Returns callback function used with `on_message_callback` of websocket.WebSocketApp.
    """

    def callback(ws_app: WebSocketApp, msg: str):
        # We strongly trust that the contract on API will hold atm :D
        event_dict = json.loads(msg)
        labels = _ClusterLogEventLabels(**event_dict["labels"])

        if "message" in event_dict:
            message = event_dict["message"]
            timestamp = dateutil.parser.isoparse(event_dict["timestamp"])
            event = _ClusterLogEvent(
                message=message,
                timestamp=timestamp,
                labels=labels,
            )
            read_queue.put(event)

    return callback


def _cluster_logs_reader(
    client: LightningClient,
    cluster_id: str,
    start: arrow.Arrow,
    end: arrow.Arrow,
    limit: int,
    follow: bool,
    on_error_callback: Optional[Callable] = None,
) -> Iterator[_ClusterLogEvent]:

    logs_api_client = _ClusterLogsSocketAPI(client.api_client)
    read_queue = queue.PriorityQueue()

    # We will use a socket inside a thread to read logs,
    # to follow our typical reading pattern
    log_socket = logs_api_client.create_cluster_logs_socket(
        cluster_id=cluster_id,
        start=start,
        end=end,
        limit=limit,
        on_message_callback=_push_log_events_to_read_queue_callback(read_queue),
        on_error_callback=on_error_callback or _error_callback,
    )

    log_thread = Thread(target=log_socket.run_forever)

    # Establish connection and begin pushing logs to the print queue
    log_thread.start()

    # Print logs from queue when log event is available
    try:
        while True:
            log_event = read_queue.get(timeout=None if follow else 1.0)
            yield log_event

    except queue.Empty:
        # Empty is raised by queue.get if timeout is reached. Follow = False case.
        pass

    except KeyboardInterrupt:
        # User pressed CTRL+C to exit, we should respect that
        pass

    finally:
        # Close connection - it will cause run_forever() to finish -> thread as finishes as well
        log_socket.close()

        # The socket was closed, we can just wait for thread to finish.
        log_thread.join()
