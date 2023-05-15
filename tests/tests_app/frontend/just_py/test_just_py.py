import os
import os.path as osp
import sys
from typing import Callable
from unittest.mock import MagicMock

import lightning.app
from lightning.app.frontend import JustPyFrontend
from lightning.app.frontend.just_py import just_py
from lightning.app.frontend.just_py.just_py_base import _main, _webpage


def render_fn(get_state: Callable) -> Callable:
    return _webpage


def test_justpy_frontend(monkeypatch):
    justpy = MagicMock()
    popen = MagicMock()
    monkeypatch.setitem(sys.modules, "justpy", justpy)
    monkeypatch.setattr(just_py, "Popen", popen)

    frontend = JustPyFrontend(render_fn=render_fn)
    flow = MagicMock()
    flow.name = "c"
    frontend.flow = flow
    frontend.start_server("a", 90)

    path = osp.join(osp.dirname(lightning.app.frontend.just_py.__file__), "just_py_base.py")

    assert popen._mock_call_args[0][0] == f"{sys.executable} {path}"
    env = popen._mock_call_args[1]["env"]
    assert env["LIGHTNING_FLOW_NAME"] == "c"
    assert env["LIGHTNING_RENDER_FUNCTION"] == "render_fn"
    assert env["LIGHTNING_HOST"] == "a"
    assert env["LIGHTNING_PORT"] == "90"

    monkeypatch.setattr(os, "environ", env)

    _main()

    assert justpy.app._mock_mock_calls[0].args[0] == "/c"
    assert justpy.app._mock_mock_calls[0].args[1] == _webpage

    assert justpy.justpy._mock_mock_calls[0].args[0] == _webpage
    assert justpy.justpy._mock_mock_calls[0].kwargs == {"host": "a", "port": 90}
