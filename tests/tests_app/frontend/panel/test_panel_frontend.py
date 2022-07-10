"""The PanelFrontend wraps your Panel code in your LightningFlow."""
# pylint: disable=protected-access, too-few-public-methods
import os
import runpy
import sys
from unittest import mock
from unittest.mock import Mock

import pytest

from lightning_app import LightningFlow
from lightning_app.frontend.panel import PanelFrontend
from lightning_app.utilities.state import AppState


def test_stop_server_not_running():
    """If the server is not running but stopped an Exception should be raised."""
    frontend = PanelFrontend(render_fn_or_file=Mock())
    with pytest.raises(RuntimeError, match="Server is not running."):
        frontend.stop_server()


def _noop_render_fn(_):
    pass


class MockFlow(LightningFlow):
    """Test Flow"""

    @property
    def name(self):
        """Return name"""
        return "root.my.flow"

    def run(self):  # pylint: disable=arguments-differ
        "Be lazy!"


@mock.patch("lightning_app.frontend.panel.panel_frontend.subprocess")
def test_panel_frontend_start_stop_server(subprocess_mock):
    """Test that `PanelFrontend.start_server()` invokes subprocess.Popen with the right
    parameters."""
    # Given
    frontend = PanelFrontend(render_fn_or_file=_noop_render_fn)
    frontend.flow = MockFlow()
    # When
    frontend.start_server(host="hostname", port=1111)
    # Then
    subprocess_mock.Popen.assert_called_once()

    env_variables = subprocess_mock.method_calls[0].kwargs["env"]
    call_args = subprocess_mock.method_calls[0].args[0]
    assert call_args[0] == sys.executable
    assert call_args[1].exists()
    assert str(call_args[1]).endswith("panel_serve_render_fn_or_file.py")
    assert len(call_args) == 2

    assert env_variables["LIGHTNING_FLOW_NAME"] == "root.my.flow"
    assert env_variables["LIGHTNING_RENDER_ADDRESS"] == "hostname"
    assert env_variables["LIGHTNING_RENDER_FUNCTION"] == "_noop_render_fn"
    assert env_variables["LIGHTNING_RENDER_MODULE_FILE"] == __file__
    assert env_variables["LIGHTNING_RENDER_PORT"] == "1111"

    assert "LIGHTNING_FLOW_NAME" not in os.environ
    assert "LIGHTNING_RENDER_FUNCTION" not in os.environ
    assert "LIGHTNING_RENDER_MODULE_FILE" not in os.environ
    assert "LIGHTNING_RENDER_MODULE_PORT" not in os.environ
    assert "LIGHTNING_RENDER_MODULE_ADDRESS" not in os.environ
    # When
    frontend.stop_server()
    # Then
    subprocess_mock.Popen().kill.assert_called_once()


def _call_me(state):
    assert isinstance(state, AppState)


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_FLOW_NAME": "root",
        "LIGHTNING_RENDER_FUNCTION": "_call_me",
        "LIGHTNING_RENDER_MODULE_FILE": __file__,
        "LIGHTNING_RENDER_ADDRESS": "127.0.0.1",
        "LIGHTNING_RENDER_PORT": "61896",
    },
)
def test_panel_wrapper_calls_render_fn_or_file(*_):
    """Run the panel_serve_render_fn_or_file"""
    runpy.run_module("lightning_app.frontend.panel.panel_serve_render_fn_or_file")
    # TODO: find a way to assert that _call_me got called


def test_method_exception():
    """The PanelFrontend does not support render_fn_or_file being a method
    and should raise an Exception"""

    class _DummyClass:
        def _render_fn(self):
            pass

    with pytest.raises(TypeError, match="being a method"):
        PanelFrontend(render_fn_or_file=_DummyClass()._render_fn)
