"""We have some utility functions that can be used across frontends."""

from lightning.app.frontend.utils import _get_flow_state, _get_frontend_environment
from lightning.app.utilities.state import AppState


def test_get_flow_state(flow_state_state: dict, flow):
    """We have a method to get an AppState scoped to the Flow state."""
    # When
    flow_state = _get_flow_state(flow)
    # Then
    assert isinstance(flow_state, AppState)
    assert flow_state._state == flow_state_state  # pylint: disable=protected-access


def some_fn(_):
    """Be lazy!"""


def test_get_frontend_environment_fn():
    """We have a utility function to get the frontend render_fn environment."""
    # When
    env = _get_frontend_environment(flow="root.lit_frontend", render_fn_or_file=some_fn, host="myhost", port=1234)
    # Then
    assert env["LIGHTNING_FLOW_NAME"] == "root.lit_frontend"
    assert env["LIGHTNING_RENDER_ADDRESS"] == "myhost"
    assert env["LIGHTNING_RENDER_FUNCTION"] == "some_fn"
    assert env["LIGHTNING_RENDER_MODULE_FILE"] == __file__
    assert env["LIGHTNING_RENDER_PORT"] == "1234"


def test_get_frontend_environment_file():
    """We have a utility function to get the frontend render_fn environment."""
    # When
    env = _get_frontend_environment(
        flow="root.lit_frontend", render_fn_or_file="app_panel.py", host="myhost", port=1234
    )
    # Then
    assert env["LIGHTNING_FLOW_NAME"] == "root.lit_frontend"
    assert env["LIGHTNING_RENDER_ADDRESS"] == "myhost"
    assert env["LIGHTNING_RENDER_FILE"] == "app_panel.py"
    assert env["LIGHTNING_RENDER_PORT"] == "1234"
