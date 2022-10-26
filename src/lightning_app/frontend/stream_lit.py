import inspect
import os
import subprocess
import sys
from typing import Callable, Optional

import lightning_app
from lightning_app.frontend.frontend import Frontend
from lightning_app.utilities.imports import requires
from lightning_app.utilities.log import get_logfile


class StreamlitFrontend(Frontend):
    """A frontend for wrapping Streamlit code in your LightingFlow.

    Return this in your `LightningFlow.configure_layout()` method if you wish to build the UI with ``streamlit``.
    To use this frontend, you must first install the `streamlit` package (if running locally):

    .. code-block:: bash

        pip install streamlit

    Arguments:
        render_fn: A function that contains your streamlit code. This function must accept exactly one argument, the
            `AppState` object which you can use to access variables in your flow (see example below).

    Example:

        In your LightningFlow, override the method `configure_layout`:

        .. code-block:: python

            class MyFlow(LightningFlow):
                def __init__(self):
                    super().__init__()
                    self.counter = 0

                def configure_layout(self):
                    return StreamlitFrontend(render_fn=my_streamlit_ui)


            # define this function anywhere you want
            # this gets called anytime the UI needs to refresh
            def my_streamlit_ui(state):
                import streamlit as st

                st.write("Hello from streamlit!")
                st.write(state.counter)
    """

    @requires("streamlit")
    def __init__(self, render_fn: Callable) -> None:
        super().__init__()

        if inspect.ismethod(render_fn):
            raise TypeError(
                "The `StreamlitFrontend` doesn't support `render_fn` being a method. Please, use a pure function."
            )

        self.render_fn = render_fn
        self._process: Optional[subprocess.Popen] = None

    def start_server(self, host: str, port: int) -> None:
        env = os.environ.copy()
        env["LIGHTNING_FLOW_NAME"] = self.flow.name
        env["LIGHTNING_RENDER_FUNCTION"] = self.render_fn.__name__
        env["LIGHTNING_RENDER_MODULE_FILE"] = inspect.getmodule(self.render_fn).__file__
        std_err_out = get_logfile("error.log")
        std_out_out = get_logfile("output.log")
        with open(std_err_out, "wb") as stderr, open(std_out_out, "wb") as stdout:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    os.path.join(os.path.dirname(lightning_app.frontend.__file__), "streamlit_base.py"),
                    "--server.address",
                    str(host),
                    "--server.port",
                    str(port),
                    "--server.baseUrlPath",
                    self.flow.name,
                    "--server.headless",
                    "true",  # do not open the browser window when running locally
                ],
                env=env,
                stdout=stdout,
                stderr=stderr,
            )

    def stop_server(self) -> None:
        if self._process is None:
            raise RuntimeError("Server is not running. Call `StreamlitFrontend.start_server()` first.")
        self._process.kill()
