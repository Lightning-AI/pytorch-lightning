"""The panel_serve_render_fn_or_file file gets run by Python to lunch a Panel Server with
Lightning.

These tests are for serving a render_file script or notebook.
"""
# pylint: disable=redefined-outer-name
import os
import pathlib
from unittest import mock

import pytest

from lightning_app.frontend.panel.panel_serve_render_fn_or_file import _serve


@pytest.fixture(scope="module")
def render_file():
    """Returns the path to a Panel app file"""
    path = pathlib.Path(__file__).parent / "app_panel.py"
    path = path.relative_to(pathlib.Path.cwd())
    return str(path)


@pytest.fixture(autouse=True, scope="module")
def mock_settings_env_vars(render_file):
    """Set the LIGHTNING environment variables."""

    with mock.patch.dict(
        os.environ,
        {
            "LIGHTNING_FLOW_NAME": "root.lit_flow",
            "LIGHTNING_RENDER_ADDRESS": "localhost",
            "LIGHTNING_RENDER_FILE": render_file,
            "LIGHTNING_RENDER_PORT": "61896",
        },
    ):
        yield


@mock.patch("panel.serve")
def test_serve(pn_serve: mock.MagicMock, render_file):
    """We can run python panel_serve_render_fn_or_file to serve the render_file."""
    _serve()
    pn_serve.assert_called_once_with(
        {"root.lit_flow": render_file},
        address="localhost",
        port=61896,
        websocket_origin="*",
        show=False,
        autoreload=False,
    )
