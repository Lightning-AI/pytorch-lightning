"""The panel_serve_render_fn_or_file file gets run by Python to lunch a Panel Server with
Lightning.

These tests are for serving a render_fn function.
"""
import os
from unittest import mock

import pytest

from lightning_app.frontend.panel.panel_serve_render_fn_or_file import (
    _get_view_fn,
    _render_fn_wrapper,
    _serve,
)
from lightning_app.frontend.utilities.app_state_watcher import AppStateWatcher


@pytest.fixture(autouse=True, scope="module")
def mock_settings_env_vars():
    """Set the LIGHTNING environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "LIGHTNING_FLOW_NAME": "root.lit_flow",
            "LIGHTNING_RENDER_ADDRESS": "localhost",
            "LIGHTNING_RENDER_FUNCTION": "render_fn",
            "LIGHTNING_RENDER_MODULE_FILE": __file__,
            "LIGHTNING_RENDER_PORT": "61896",
        },
    ):
        yield


def render_fn(app):
    """Test function that just passes through the app."""
    return app


def test_get_view_fn():
    """We have a helper get_view_fn function that create a function for our view.

    If the render_fn provides an argument an AppStateWatcher is provided as argument
    """
    view_fn = _get_view_fn()
    result = view_fn()
    assert isinstance(result, AppStateWatcher)


def render_fn_no_args():
    """Test function with no arguments"""
    return "Hello"


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_RENDER_FUNCTION": "render_fn_no_args",
        "LIGHTNING_RENDER_MODULE_FILE": __file__,
    },
)
def test_get_view_fn_no_args():
    """We have a helper get_view_fn function that create a function for our view.

    If the render_fn provides an argument an AppStateWatcher is provided as argument
    """
    view_fn = _get_view_fn()
    result = view_fn()
    assert result == "Hello"


@mock.patch("panel.serve")
def test_serve(pn_serve: mock.MagicMock):
    """We can run python panel_serve_render_fn_or_file to serve the render_fn."""
    _serve()
    pn_serve.assert_called_once_with(
        {"root.lit_flow": _render_fn_wrapper},
        address="localhost",
        port=61896,
        websocket_origin="*",
        show=False,
        autoreload=False,
    )
