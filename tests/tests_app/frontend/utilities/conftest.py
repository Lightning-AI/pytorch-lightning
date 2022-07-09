import os
from unittest import mock

import pytest

FLOW_SUB = "lit_panel"
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


def render_fn(app):
    """Test function that just passes through the app."""
    return app


@pytest.fixture(autouse=True, scope="module")
def mock_request_state():
    """Avoid requests to the api."""
    with mock.patch("lightning_app.utilities.state.AppState._request_state", _request_state):
        yield


@pytest.fixture(autouse=True, scope="module")
def mock_settings_env_vars():
    """Set the LIGHTNING environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "LIGHTNING_FLOW_NAME": FLOW,
            "LIGHTNING_RENDER_ADDRESS": "localhost",
            "LIGHTNING_RENDER_FUNCTION": "render_fn",
            "LIGHTNING_RENDER_MODULE_FILE": __file__,
            "LIGHTNING_RENDER_PORT": f"{PORT}",
        },
    ):
        yield


def do_nothing():
    """Be lazy!"""


@pytest.fixture(autouse=True, scope="module")
def mock_start_websocket():
    """Avoid starting the websocket."""
    with mock.patch("lightning_app.frontend.utilities.app_state_comm._start_websocket", do_nothing):
        yield


@pytest.fixture
def app_state_state():
    return APP_STATE.copy()


@pytest.fixture
def flow_state_state():
    return FLOW_STATE.copy()
