"""We have some utility functions that can be used across frontends"""
import inspect
import os

from lightning_app.frontend.utilities.other import (
    get_flow_state,
    get_frontend_environment,
    get_render_fn_from_environment,
)
from lightning_app.utilities.state import AppState


def test_get_flow_state(flow_state_state: dict):
    """We have a method to get an AppState scoped to the Flow state"""
    # When
    flow_state = get_flow_state()
    # Then
    assert isinstance(flow_state, AppState)
    assert flow_state._state == flow_state_state # pylint: disable=protected-access


def test_get_render_fn_from_environment():
    """We have a method to get the render_fn from the environment"""
    # When
    render_fn = get_render_fn_from_environment()
    # Then
    assert inspect.getfile(render_fn) == os.environ["LIGHTNING_RENDER_MODULE_FILE"]
    assert render_fn.__name__ == os.environ["LIGHTNING_RENDER_FUNCTION"]


def some_fn(_):
    """Be lazy!"""


def test__get_frontend_environment():
    """We have a utility function to get the frontend render_fn environment"""
    # When
    env = get_frontend_environment(
        flow="root.lit_frontend", render_fn=some_fn, host="myhost", port=1234
    )
    # Then
    assert env["LIGHTNING_FLOW_NAME"] == "root.lit_frontend"
    assert env["LIGHTNING_RENDER_ADDRESS"] == "myhost"
    assert env["LIGHTNING_RENDER_FUNCTION"] == "some_fn"
    assert env["LIGHTNING_RENDER_MODULE_FILE"] == __file__
    assert env["LIGHTNING_RENDER_PORT"] == "1234"
