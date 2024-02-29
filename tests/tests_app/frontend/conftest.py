"""Test configuration."""

# pylint: disable=protected-access
from unittest import mock

import pytest

FLOW_SUB = "lit_flow"
FLOW = f"root.{FLOW_SUB}"
PORT = 61896

FLOW_STATE = {
    "vars": {
        "_paths": {},
        "_layout": {"target": f"http://localhost:{PORT}/{FLOW}"},
    },
    "calls": {},
    "flows": {},
    "works": {},
    "structures": {},
    "changes": {},
}

APP_STATE = {
    "vars": {"_paths": {}, "_layout": [{"name": "home", "content": FLOW}]},
    "calls": {},
    "flows": {
        FLOW_SUB: FLOW_STATE,
    },
    "works": {},
    "structures": {},
    "changes": {},
    "app_state": {"stage": "running"},
}


def _request_state(self):
    _state = APP_STATE
    self._store_state(_state)


@pytest.fixture()
def flow():
    return FLOW


@pytest.fixture(autouse=True, scope="module")
def mock_request_state():
    """Avoid requests to the api."""
    with mock.patch("lightning.app.utilities.state.AppState._request_state", _request_state):
        yield


def do_nothing():
    """Be lazy!"""


@pytest.fixture(autouse=True, scope="module")
def mock_start_websocket():
    """Avoid starting the websocket."""
    with mock.patch("lightning.app.frontend.panel.app_state_comm._start_websocket", do_nothing):
        yield


@pytest.fixture()
def app_state_state():
    """Returns an AppState dict."""
    return APP_STATE.copy()


@pytest.fixture()
def flow_state_state():
    """Returns an AppState dict scoped to the flow."""
    return FLOW_STATE.copy()
