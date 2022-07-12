"""The PanelFrontend wraps your Panel code in your LightningFlow."""
from __future__ import annotations

import inspect
import logging
import pathlib
import subprocess
import sys
from typing import Callable, Dict, List

from other import get_allowed_hosts, get_frontend_environment, has_panel_autoreload, is_running_locally

from lightning_app.frontend.frontend import Frontend
from lightning_app.utilities.imports import requires
from lightning_app.utilities.log import get_frontend_logfile

_logger = logging.getLogger("PanelFrontend")


class PanelFrontend(Frontend):
    # Todo: Add Example to docstring
    """The PanelFrontend enables you to serve Panel code as a Frontend for your LightningFlow.

    To use this frontend, you must first install the `panel` package:

    .. code-block:: bash

        pip install panel

    Please note the Panel server will be logging output to error.log and output.log files
    respectively.

    You can start the lightning server with Panel autoreload by setting the `PANEL_AUTORELOAD`
    environment variable to 'yes': `AUTORELOAD=yes lightning run app my_app.py`.

    Args:
        render_fn_or_file: A pure function or the path to a .py or .ipynb file.
        The function must be a pure function that contains your Panel code.
        The function can optionally accept an `AppStateWatcher` argument.

    Raises:
        TypeError: Raised if the render_fn_or_file is a class method
    """

    @requires("panel")
    def __init__(self, render_fn_or_file: Callable | str):
        # Todo: consider renaming back to render_fn or something else short.
        # Its a hazzle reading and writing such a long name
        super().__init__()

        if inspect.ismethod(render_fn_or_file):
            raise TypeError(
                "The `PanelFrontend` doesn't support `render_fn_or_file` being a method. "
                "Please, use a pure function."
            )

        self.render_fn_or_file = render_fn_or_file
        self._process: None | subprocess.Popen = None
        self._log_files: Dict[str] = {}
        _logger.debug("initialized")

    def _get_popen_args(self, host: str, port: int) -> List:
        if callable(self.render_fn_or_file):
            path = str(pathlib.Path(__file__).parent / "panel_serve_render_fn.py")
        else:
            path = pathlib.Path(self.render_fn_or_file)

        abs_path = str(path)
        # The app is served at http://localhost:{port}/{flow}/{render_fn_or_file}
        # Lightning embeds http://localhost:{port}/{flow} but this redirects to the above and
        # seems to work fine.
        command = [
            sys.executable,
            "-m",
            "panel",
            "serve",
            abs_path,
            "--port",
            str(port),
            "--address",
            host,
            "--prefix",
            self.flow.name,
            "--allow-websocket-origin",
            get_allowed_hosts(),
        ]
        if has_panel_autoreload():
            command.append("--autoreload")
        _logger.debug("%s", command)
        return command

    def start_server(self, host: str, port: int) -> None:
        _logger.debug("starting server %s %s", host, port)
        env = get_frontend_environment(
            self.flow.name,
            self.render_fn_or_file,
            port,
            host,
        )
        command = self._get_popen_args(host, port)

        if not is_running_locally():
            self._open_log_files()

        self._process = subprocess.Popen(  # pylint: disable=consider-using-with
            command, env=env, **self._log_files
        )

    def stop_server(self) -> None:
        if self._process is None:
            raise RuntimeError("Server is not running. Call `PanelFrontend.start_server()` first.")
        self._process.kill()
        self._close_log_files()

    def _close_log_files(self):
        for file_ in self._log_files.values():
            if not file_.closed:
                file_.close()
        self._log_files = {}

    def _open_log_files(self) -> Dict:
        # Don't log to file when developing locally. Makes it harder to debug.
        self._close_log_files()

        std_err_out = get_frontend_logfile("error.log")
        std_out_out = get_frontend_logfile("output.log")
        stderr = std_err_out.open("wb")
        stdout = std_out_out.open("wb")
        self._log_files = {"stdout": stderr, "stderr": stdout}
