"""The AppStateWatcher enables a Frontend to.

- subscribe to app state changes
- to access and change the app state.

This is particularly useful for the PanelFrontend but can be used by other Frontends too.
"""
# pylint: disable=protected-access
from lightning_app.frontend.utilities.app_state_watcher import AppStateWatcher
from lightning_app.utilities.state import AppState


def test_init(flow_state_state: dict):
    """We can instantiate the AppStateWatcher.

    - the .state is set
    - the .state is scoped to the flow state
    """
    app = AppStateWatcher()
    assert isinstance(app.state, AppState)
    assert app.state._state == flow_state_state


def test_handle_state_changed(flow_state_state: dict):
    """We can handle state changes by updating the state."""
    app = AppStateWatcher()
    org_state = app.state
    app._handle_state_changed()
    assert app.state is not org_state
    assert app.state._state == flow_state_state


def test_is_singleton():
    """The AppStateWatcher is a singleton for efficiency reasons."""
    watcher1 = AppStateWatcher()
    watcher2 = AppStateWatcher()
    assert watcher1 is watcher2
