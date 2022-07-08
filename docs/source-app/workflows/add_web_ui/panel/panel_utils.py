from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Callable

import param
import websockets

from lightning_app.core.constants import APP_SERVER_PORT
from lightning_app.frontend.streamlit_base import _app_state_to_flow_scope
from lightning_app.utilities.state import AppState

logger = logging.getLogger("PanelFrontend")

_CALLBACKS = []
_THREAD: None | threading.Thread = None

def _target_fn():
    async def update_fn():
        url = "localhost:8080" if "LIGHTNING_APP_STATE_URL" in os.environ else f"localhost:{APP_SERVER_PORT}"
        ws_url = f"ws://{url}/api/v1/ws"
        logger.debug("connecting to web socket %s", ws_url)
        async with websockets.connect(ws_url) as websocket:
            while True:
                await websocket.recv()
                # while (time.time() - last_updated) < 0.2:
                #     time.sleep(0.05)
                logger.debug("App State Changed. Running callbacks")
                for callback in _CALLBACKS:
                    callback()
                

    asyncio.run(update_fn())

def _start_websocket():
    global _THREAD
    if not _THREAD:
        logger.debug("starting thread")
        _THREAD = threading.Thread(target=_target_fn)
        _THREAD.setDaemon(True)
        _THREAD.start()
        logger.debug("thread started")

def watch_app_state(callback: Callable):
    _CALLBACKS.append(callback)
    
    _start_websocket()

def get_flow_state():
    app_state = AppState()
    app_state._request_state()
    flow = os.environ["LIGHTNING_FLOW_NAME"]
    flow_state = _app_state_to_flow_scope(app_state, flow)
    return flow_state

class AppStateWatcher(param.Parameterized):
    state: AppState = param.ClassSelector(class_=AppState)
    
    def __init__(self):
        app_state = self._get_flow_state()
        super().__init__(state=app_state)
        watch_app_state(self.handle_state_changed)

    def _get_flow_state(self):
        return get_flow_state()

    def _request_state(self):
        self.state = self._get_flow_state()
        logger.debug("Request app state")

    def handle_state_changed(self):
        logger.debug("Handle app state changed")
        self._request_state()
