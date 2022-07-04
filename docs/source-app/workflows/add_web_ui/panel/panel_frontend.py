import logging
import sys

import panel as pn

import lightning_app as lapp
from lightning_app.utilities.imports import requires

logger = logging.getLogger("PanelFrontend")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class PanelFrontend(lapp.LightningWork):
    @requires("panel")
    def __init__(self, render_fn, parallel=True, **kwargs):
        super().__init__(parallel=parallel, **kwargs)
        self._render_fn = render_fn
        self._server = None
        self.requests = 0
        logger.info("init finished")

    def _fast_initial_view(self):
        self.requests += 1
        logger.info("Session %s started", self.requests)
        if self.requests == 1:
            return pn.pane.HTML("<h1>Please refresh the browser to see the app.</h1>")
        else:
            return self._render_fn(self)

    def run(self):
        logger.info("run start")
        if not self._server:
            logger.info("LitPanel Starting Server")
            self._server = pn.serve(
                {"/": self._fast_initial_view},
                port=self.port,
                address=self.host,
                websocket_origin="*",
                show=False,
                # threaded=True,
            )
        logger.info("run end")

    def stop(self):
        """Stops the server."""
        if self._server:
            self._server.stop()
        logger.info("stop end")
