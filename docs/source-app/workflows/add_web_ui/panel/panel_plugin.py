import asyncio
import logging
import os
import sys
import threading
import time

import param
import websockets

from lightning_app.core.constants import APP_SERVER_PORT
from lightning_app.utilities.app_helpers import AppStateType, BaseStatePlugin

logger = logging.getLogger("PanelPlugin")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ParamState(param.Parameterized):
    value: int = param.Integer()

def target_fn(param_state: ParamState):
    async def update_fn(param_state: ParamState):
        url = "localhost:8080" if "LIGHTNING_APP_STATE_URL" in os.environ else f"localhost:{APP_SERVER_PORT}"
        ws_url = f"ws://{url}/api/v1/ws"
        last_updated = time.time()
        logger.info("connecting to web sockets %s", ws_url)
        async with websockets.connect(ws_url) as websocket:
            while True:
                await websocket.recv()
                while (time.time() - last_updated) < 1:
                    time.sleep(0.1)
                last_updated = time.time()
                param_state.value += 1
                logger.info("App state changed")

    asyncio.run(update_fn(param_state))



class PanelStatePlugin(BaseStatePlugin):
    def __init__(self):
        super().__init__()
        import panel as pn
        self.param_state = ParamState(value=1)

        if "_lightning_websocket_thread" not in pn.state.cache:
            logger.info("starting thread")
            thread = threading.Thread(target=target_fn, args=(self.param_state, ))
            pn.state.cache["_lightning_websocket_thread"] = thread
            thread.setDaemon(True)
            thread.start()
            logger.info("thread started")

    def should_update_app(self, deep_diff):
        return deep_diff

    def get_context(self):
        return {"type": AppStateType.DEFAULT.value}

    def render_non_authorized(self):
        pass
