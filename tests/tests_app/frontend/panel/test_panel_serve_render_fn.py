"""This panel_serve_render_fn file gets run by Python to lunch a Panel Server with Lightning."""
import os
from unittest import mock

import pytest

from lightning_app.frontend.panel.panel_serve_render_fn import _serve, _view_fn
from lightning_app.frontend.utilities.app_state_watcher import AppStateWatcher


@pytest.fixture(autouse=True, scope="module")
def mock_settings_env_vars():
    """Set the LIGHTNING environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "LIGHTNING_FLOW_NAME": "root.lit_panel",
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


def test_view():
    """We have a helper _view function that provides the AppStateWatcher as argument to render_fn and returns the
    result."""
    result = _view_fn()
    assert isinstance(result, AppStateWatcher)


@mock.patch("panel.serve")
def test_serve(pn_serve: mock.MagicMock):
    """We can run python panel_serve_render_fn to serve the render_fn."""
    _serve()
    pn_serve.assert_called_once_with(
        {"root.lit_panel": _view_fn}, address="localhost", port=61896, websocket_origin="*", show=False
    )
