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

"""The PanelFrontend wraps your Panel code in your LightningFlow."""
from __future__ import annotations

import inspect
import os
import pathlib
import subprocess
import sys
from typing import Callable, TextIO

from lightning_app.frontend.frontend import Frontend
from lightning_app.frontend.utils import _get_frontend_environment
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.cloud import is_running_in_cloud
from lightning_app.utilities.imports import requires
from lightning_app.utilities.log import get_logfile

_logger = Logger(__name__)


def _has_panel_autoreload() -> bool:
    """Returns True if the PANEL_AUTORELOAD environment variable is set to 'yes' or 'true'.

    Please note the casing of value does not matter
    """
    return os.environ.get("PANEL_AUTORELOAD", "no").lower() in ["yes", "y", "true"]


class PanelFrontend(Frontend):
    """The `PanelFrontend` enables you to serve Panel code as a Frontend for your LightningFlow.

    Reference: https://lightning.ai/lightning-docs/workflows/add_web_ui/panel/

    Args:
        entry_point: The path to a .py or .ipynb file, or a pure function. The file or function must contain your Panel
            code. The function can optionally accept an ``AppStateWatcher`` argument.

    Raises:
        TypeError: Raised if the ``entry_point`` provided is a class method

    Example:

    To use the `PanelFrontend`, you must first install the `panel` package:

    .. code-block:: bash

        pip install panel

    Create the files `panel_app_basic.py` and `app_basic.py` with the content below.

    **panel_app_basic.py**

    .. code-block:: python

        import panel as pn

        pn.panel("Hello **Panel ⚡** World").servable()

    **app_basic.py**

    .. code-block:: python

        import lightning as L
        from lightning_app.frontend.panel import PanelFrontend


        class LitPanel(L.LightningFlow):
            def configure_layout(self):
                return PanelFrontend("panel_app_basic.py")


        class LitApp(L.LightningFlow):
            def __init__(self):
                super().__init__()
                self.lit_panel = LitPanel()

            def configure_layout(self):
                return {"name": "home", "content": self.lit_panel}


        app = L.LightningApp(LitApp())

    Start the Lightning server with `lightning run app app_basic.py`.

    For development you can get Panel autoreload by setting the ``PANEL_AUTORELOAD``
    environment variable to 'yes', i.e. run
    ``PANEL_AUTORELOAD=yes lightning run app app_basic.py``
    """

    @requires("panel")
    def __init__(self, entry_point: str | Callable):
        super().__init__()

        if inspect.ismethod(entry_point):
            raise TypeError(
                "The `PanelFrontend` doesn't support `entry_point` being a method. Please, use a pure function."
            )

        self.entry_point = entry_point
        self._process: None | subprocess.Popen = None
        self._log_files: dict[str, TextIO] = {}
        _logger.debug("PanelFrontend Frontend with %s is initialized.", entry_point)

    def start_server(self, host: str, port: int, root_path: str = "") -> None:
        _logger.debug("PanelFrontend starting server on %s:%s", host, port)

        # 1: Prepare environment variables and arguments.
        env = _get_frontend_environment(
            self.flow.name,
            self.entry_point,
            port,
            host,
        )
        command = self._get_popen_args(host, port)

        if is_running_in_cloud():
            self._open_log_files()

        self._process = subprocess.Popen(command, env=env, **self._log_files)  # pylint: disable=consider-using-with

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

    def _open_log_files(self) -> None:
        # Don't log to file when developing locally. Makes it harder to debug.
        self._close_log_files()

        std_err_out = get_logfile("error.log")
        std_out_out = get_logfile("output.log")
        stderr = std_err_out.open("wb")
        stdout = std_out_out.open("wb")
        self._log_files = {"stdout": stderr, "stderr": stdout}

    def _get_popen_args(self, host: str, port: int) -> list:
        if callable(self.entry_point):
            path = str(pathlib.Path(__file__).parent / "panel_serve_render_fn.py")
        else:
            path = pathlib.Path(self.entry_point)

        abs_path = str(path)
        # The app is served at http://localhost:{port}/{flow}/{entry_point}
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
            _get_allowed_hosts(),
        ]
        if _has_panel_autoreload():
            command.append("--autoreload")
        _logger.debug("PanelFrontend command %s", command)
        return command


def _get_allowed_hosts() -> str:
    """Returns a comma separated list of host[:port] that should be allowed to connect."""
    # TODO: Enable only lightning.ai domain in the cloud
    return "*"
