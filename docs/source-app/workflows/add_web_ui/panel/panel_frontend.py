import inspect
import logging
import os
import pathlib
import subprocess
import sys
from typing import Callable

from lightning_app.frontend.frontend import Frontend
from lightning_app.utilities.imports import requires
from lightning_app.utilities.log import get_frontend_logfile

logger = logging.getLogger("PanelFrontend")


class PanelFrontend(Frontend):
    @requires("panel")
    def __init__(self, render_fn: Callable):  # Would like to accept a `render_file` arguemnt too in the future
        super().__init__()

        if inspect.ismethod(render_fn):
            raise TypeError(
                "The `PanelFrontend` doesn't support `render_fn` being a method. Please, use a pure function."
            )

        self.render_fn = render_fn
        logger.debug("initialized")

    def start_server(self, host: str, port: int) -> None:
        logger.debug("starting server %s %s", host, port)
        env = os.environ.copy()
        env["LIGHTNING_FLOW_NAME"] = self.flow.name
        env["LIGHTNING_RENDER_FUNCTION"] = self.render_fn.__name__
        env["LIGHTNING_RENDER_MODULE_FILE"] = inspect.getmodule(self.render_fn).__file__
        env["LIGHTNING_RENDER_PORT"] = str(port)
        env["LIGHTNING_RENDER_ADDRESS"] = str(host)
        std_err_out = get_frontend_logfile("error.log")
        std_out_out = get_frontend_logfile("output.log")
        with open(std_err_out, "wb") as stderr, open(std_out_out, "wb") as stdout:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    pathlib.Path(__file__).parent / "panel_serve.py",
                ],
                env=env,
                # stdout=stdout,
                # stderr=stderr,
            )

    def stop_server(self) -> None:
        if self._process is None:
            raise RuntimeError("Server is not running. Call `PanelFrontend.start_server()` first.")
        self._process.kill()
