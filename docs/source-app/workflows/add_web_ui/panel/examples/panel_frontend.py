"""The PanelFrontend wraps your Panel code in your LightningFlow."""
from __future__ import annotations

import inspect
import logging
import pathlib
import subprocess
import sys
from typing import Callable

from lightning_app.frontend.frontend import Frontend
from other import get_frontend_environment
from lightning_app.utilities.imports import requires
from lightning_app.utilities.log import get_frontend_logfile

_logger = logging.getLogger("PanelFrontend")


class PanelFrontend(Frontend):
    """The PanelFrontend enables you to serve Panel code as a Frontend for your LightningFlow.

    To use this frontend, you must first install the `panel` package:

    .. code-block:: bash

        pip install panel

    Please note the Panel server will be logging output to error.log and output.log files
    respectively.

    # Todo: Add Example

    Args:
        render_fn: A pure function that contains your Panel code. This function must accept
            exactly one argument, the `AppStateWatcher` object which you can use to get and
            set variables in your flow (see example below). This function must return a
            Panel Viewable.

    Raises:
        TypeError: Raised if the render_fn is a class method
    """

    @requires("panel")
    def __init__(self, render_fn: Callable):
        # Todo: enable the render_fn to be a .py or .ipynb file
        # Todo: enable the render_fn to not accept an AppStateWatcher as argument
        super().__init__()

        if inspect.ismethod(render_fn):
            raise TypeError(
                "The `PanelFrontend` doesn't support `render_fn` being a method. Please, use a " "pure function."
            )

        self.render_fn = render_fn
        self._process: None | subprocess.Popen = None
        _logger.debug("initialized")

    def start_server(self, host: str, port: int) -> None:
        _logger.debug("starting server %s %s", host, port)
        env = get_frontend_environment(
            self.flow.name,
            self.render_fn,
            port,
            host,
        )
        std_err_out = get_frontend_logfile("error.log")
        std_out_out = get_frontend_logfile("output.log")
        with open(std_err_out, "wb") as stderr, open(std_out_out, "wb") as stdout:
            self._process = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    sys.executable,
                    pathlib.Path(__file__).parent / "panel_serve_render_fn.py",
                ],
                env=env,
                # stdout=stdout,
                # stderr=stderr,
            )

    def stop_server(self) -> None:
        if self._process is None:
            raise RuntimeError("Server is not running. Call `PanelFrontend.start_server()` first.")
        self._process.kill()
