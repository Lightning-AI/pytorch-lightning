import inspect
import os
import sys
from subprocess import Popen
from time import sleep
from typing import Callable, Optional

import lightning_app
from lightning_app.frontend.frontend import Frontend
from lightning_app.utilities.log import get_frontend_logfile


class JustPyFrontend(Frontend):
    def __init__(self, render_fn: Callable) -> None:
        super().__init__()

        if inspect.ismethod(render_fn):
            raise TypeError(
                "The `StreamlitFrontend` doesn't support `render_fn` being a method. Please, use a pure function."
            )

        self.render_fn = render_fn
        self._process: Optional[Popen] = None

    def start_server(self, host: str, port: int) -> None:
        env = os.environ.copy()
        env["LIGHTNING_FLOW_NAME"] = self.flow.name
        env["LIGHTNING_RENDER_FUNCTION"] = self.render_fn.__name__
        env["LIGHTNING_RENDER_MODULE_FILE"] = inspect.getmodule(self.render_fn).__file__
        env["LIGHTNING_HOST"] = host
        env["LIGHTNING_PORT"] = str(port)
        std_out_out = get_frontend_logfile("output.log")
        path = os.path.join(os.path.dirname(lightning_app.frontend.just_py.__file__), "just_py_base.py")
        with open(std_out_out, "wb") as stdout:
            self._process = Popen(f"{sys.executable} {path}", env=env, stdout=stdout, stderr=sys.stderr, shell=True)

        sleep(1)

    def stop_server(self) -> None:
        assert self._process
        self._process.terminate()
