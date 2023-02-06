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

"""The watch_app_state function enables us to trigger a callback function when ever the app state changes."""
# Todo: Refactor with Streamlit
# Note: It would be nice one day to just watch changes within the Flow scope instead of whole app
from __future__ import annotations

import asyncio
import os
from threading import Thread
from typing import Callable

import websockets

from lightning_app.core.constants import APP_SERVER_PORT
from lightning_app.utilities.app_helpers import Logger

_logger = Logger(__name__)

_CALLBACKS = []
_THREAD: Thread = None


def _get_ws_port():
    if "LIGHTNING_APP_STATE_URL" in os.environ:
        return 8080
    return APP_SERVER_PORT


def _get_ws_url():
    port = _get_ws_port()
    return f"ws://localhost:{port}/api/v1/ws"


def _run_callbacks():
    for callback in _CALLBACKS:
        callback()


def _target_fn():
    async def update_fn():
        ws_url = _get_ws_url()
        _logger.debug("connecting to web socket %s", ws_url)
        async with websockets.connect(ws_url) as websocket:  # pylint: disable=no-member
            while True:
                await websocket.recv()
                # Note: I have not seen use cases where the two lines below are needed
                # Changing '< 0.2' to '< 1' makes the App very sluggish to the end user
                # Also the implementation can cause the App state to lag behind because only 1 update
                # is received per 0.2 second (or 1 second).
                # while (time.time() - last_updated) < 0.2:
                #     time.sleep(0.05)

                # Todo: Add some kind of throttling. If 10 messages are received within 100ms then
                # there is no need to trigger the app state changed, request state and update
                # 10 times.
                _logger.debug("App State Changed. Running callbacks")
                _run_callbacks()

    asyncio.run(update_fn())


def _start_websocket():
    global _THREAD  # pylint: disable=global-statement
    if not _THREAD:
        _logger.debug("Starting the watch_app_state thread.")
        _THREAD = Thread(target=_target_fn)
        _THREAD.setDaemon(True)
        _THREAD.start()
        _logger.debug("thread started")


def _watch_app_state(callback: Callable):
    """Start the process that serves the UI at the given hostname and port number.

    Arguments:
        callback: A function to run when the App state changes. Must be thread safe.

    Example:

        .. code-block:: python

            def handle_state_change():
                print("The App State changed.")
                watch_app_state(handle_state_change)
    """
    _CALLBACKS.append(callback)
    _start_websocket()
