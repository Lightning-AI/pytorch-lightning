import os
import runpy
import sys
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
from lightning.app import LightningFlow
from lightning.app.frontend.stream_lit import StreamlitFrontend
from lightning.app.utilities.state import AppState
from lightning_utilities.core.imports import RequirementCache

_STREAMLIT_AVAILABLE = RequirementCache("streamlit")


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="requires streamlit")
def test_stop_server_not_running():
    frontend = StreamlitFrontend(render_fn=Mock())
    with pytest.raises(RuntimeError, match="Server is not running."):
        frontend.stop_server()


def _noop_render_fn(_):
    pass


class MockFlow(LightningFlow):
    @property
    def name(self):
        return "root.my.flow"

    def run(self):
        pass


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="requires streamlit")
@mock.patch("lightning.app.frontend.stream_lit.subprocess")
def test_streamlit_frontend_start_stop_server(subprocess_mock):
    """Test that `StreamlitFrontend.start_server()` invokes subprocess.Popen with the right parameters."""
    frontend = StreamlitFrontend(render_fn=_noop_render_fn)
    frontend.flow = MockFlow()
    frontend.start_server(host="hostname", port=1111)
    subprocess_mock.Popen.assert_called_once()

    env_variables = subprocess_mock.method_calls[0].kwargs["env"]
    call_args = subprocess_mock.method_calls[0].args[0]
    assert call_args == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        ANY,
        "--server.address",
        "hostname",
        "--server.port",
        "1111",
        "--server.baseUrlPath",
        "root.my.flow",
        "--server.headless",
        "true",
        "--server.enableXsrfProtection",
        "false",
    ]

    assert env_variables["LIGHTNING_FLOW_NAME"] == "root.my.flow"
    assert env_variables["LIGHTNING_RENDER_FUNCTION"] == "_noop_render_fn"
    assert env_variables["LIGHTNING_RENDER_MODULE_FILE"] == __file__

    assert "LIGHTNING_FLOW_NAME" not in os.environ
    assert "LIGHTNING_RENDER_FUNCTION" not in os.environ
    assert "LIGHTNING_RENDER_MODULE_FILE" not in os.environ

    frontend.stop_server()
    subprocess_mock.Popen().kill.assert_called_once()


def _streamlit_call_me(state):
    assert isinstance(state, AppState)


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_FLOW_NAME": "root",
        "LIGHTNING_RENDER_FUNCTION": "_streamlit_call_me",
        "LIGHTNING_RENDER_MODULE_FILE": __file__,
    },
)
def test_streamlit_wrapper_calls_render_fn(*_):
    runpy.run_module("lightning.app.frontend.streamlit_base")
    # TODO: find a way to assert that _streamlit_call_me got called


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="requires streamlit")
def test_method_exception():
    class A:
        def render_fn(self):
            pass

    with pytest.raises(TypeError, match="being a method"):
        StreamlitFrontend(render_fn=A().render_fn)
