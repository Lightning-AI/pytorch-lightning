"""The watch_app_state function enables us to trigger a callback function whenever the App state changes."""

import os
from unittest import mock

from lightning.app.core.constants import APP_SERVER_PORT
from lightning.app.frontend.panel.app_state_comm import _get_ws_url, _run_callbacks, _watch_app_state

FLOW_SUB = "lit_flow"
FLOW = f"root.{FLOW_SUB}"


def do_nothing():
    """Be lazy!"""


def test_get_ws_url_when_local():
    """The websocket uses port APP_SERVER_PORT when local."""
    assert _get_ws_url() == f"ws://localhost:{APP_SERVER_PORT}/api/v1/ws"


@mock.patch.dict(os.environ, {"LIGHTNING_APP_STATE_URL": "some_url"})
def test_get_ws_url_when_cloud():
    """The websocket uses port 8080 when LIGHTNING_APP_STATE_URL is set."""
    assert _get_ws_url() == "ws://localhost:8080/api/v1/ws"


@mock.patch.dict(os.environ, {"LIGHTNING_FLOW_NAME": "FLOW"})
def test_watch_app_state():
    """We can watch the App state and a callback function will be run when it changes."""
    callback = mock.MagicMock()
    # When
    _watch_app_state(callback)

    # Here we would like to send messages using the web socket
    # For testing the web socket is not started. See conftest.py
    # So we need to manually trigger _run_callbacks here
    _run_callbacks()
    # Then
    callback.assert_called_once()
