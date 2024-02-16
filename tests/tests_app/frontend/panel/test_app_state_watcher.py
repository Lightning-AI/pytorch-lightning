"""The AppStateWatcher enables a Frontend to.

- subscribe to App state changes.
- to access and change the App state.

This is particularly useful for the PanelFrontend, but can be used by other Frontends too.

"""

# pylint: disable=protected-access
import os
from unittest import mock

import pytest
from lightning.app.frontend.panel.app_state_watcher import AppStateWatcher
from lightning.app.utilities.state import AppState
from lightning_utilities.core.imports import RequirementCache

_PARAM_AVAILABLE = RequirementCache("param")

FLOW_SUB = "lit_flow"
FLOW = f"root.{FLOW_SUB}"
PORT = 61896


@pytest.fixture(autouse=True)
def mock_settings_env_vars():
    """Set the LIGHTNING environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "LIGHTNING_FLOW_NAME": FLOW,
            "LIGHTNING_RENDER_ADDRESS": "localhost",
            "LIGHTNING_RENDER_PORT": f"{PORT}",
        },
    ):
        yield


@pytest.mark.skipif(not _PARAM_AVAILABLE, reason="requires param")
def test_init(flow_state_state: dict):
    """We can instantiate the AppStateWatcher.

    - the .state is set
    - the .state is scoped to the flow state

    """
    # When
    app = AppStateWatcher()
    # Needed as AppStateWatcher is singleton and might have been
    # instantiated and the state changed in other tests
    app._update_flow_state()

    # Then
    assert isinstance(app.state, AppState)
    assert app.state._state == flow_state_state


@pytest.mark.skipif(not _PARAM_AVAILABLE, reason="requires param")
def test_update_flow_state(flow_state_state: dict):
    """We can update the state.

    - the .state is scoped to the flow state

    """
    app = AppStateWatcher()
    org_state = app.state
    app._update_flow_state()
    assert app.state is not org_state
    assert app.state._state == flow_state_state


@pytest.mark.skipif(not _PARAM_AVAILABLE, reason="requires param")
def test_is_singleton():
    """The AppStateWatcher is a singleton for efficiency reasons.

    Its key that __new__ and __init__ of AppStateWatcher is only called once. See
    https://github.com/holoviz/param/issues/643

    """
    # When
    app1 = AppStateWatcher()
    name1 = app1.name
    state1 = app1.state

    app2 = AppStateWatcher()
    name2 = app2.name
    state2 = app2.state

    # Then
    assert app1 is app2
    assert name1 == name2
    assert app1.name == name2
    assert state1 is state2
    assert app1.state is state2
