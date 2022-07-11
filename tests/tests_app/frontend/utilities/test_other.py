"""We have some utility functions that can be used across frontends."""
import inspect
import os
from unittest import mock

import pytest

from lightning_app.frontend.utilities.other import (
    get_flow_state,
    get_frontend_environment,
    get_render_fn_from_environment,
    has_panel_autoreload,
)
from lightning_app.utilities.state import AppState


def test_get_flow_state(flow_state_state: dict, flow):
    """We have a method to get an AppState scoped to the Flow state."""
    # When
    flow_state = get_flow_state(flow)
    # Then
    assert isinstance(flow_state, AppState)
    assert flow_state._state == flow_state_state  # pylint: disable=protected-access


def render_fn():
    """Do nothing"""


def test_get_render_fn_from_environment():
    """We have a method to get the render_fn from the environment."""
    # When
    result = get_render_fn_from_environment("render_fn", __file__)
    # Then
    assert result.__name__ == render_fn.__name__
    assert inspect.getmodule(result).__file__ == __file__


def some_fn(_):
    """Be lazy!"""


def test_get_frontend_environment_fn():
    """We have a utility function to get the frontend render_fn environment."""
    # When
    env = get_frontend_environment(
        flow="root.lit_frontend", render_fn_or_file=some_fn, host="myhost", port=1234
    )
    # Then
    assert env["LIGHTNING_FLOW_NAME"] == "root.lit_frontend"
    assert env["LIGHTNING_RENDER_ADDRESS"] == "myhost"
    assert env["LIGHTNING_RENDER_FUNCTION"] == "some_fn"
    assert env["LIGHTNING_RENDER_MODULE_FILE"] == __file__
    assert env["LIGHTNING_RENDER_PORT"] == "1234"


def test_get_frontend_environment_file():
    """We have a utility function to get the frontend render_fn environment."""
    # When
    env = get_frontend_environment(
        flow="root.lit_frontend", render_fn_or_file="app_panel.py", host="myhost", port=1234
    )
    # Then
    assert env["LIGHTNING_FLOW_NAME"] == "root.lit_frontend"
    assert env["LIGHTNING_RENDER_ADDRESS"] == "myhost"
    assert env["LIGHTNING_RENDER_FILE"] == "app_panel.py"
    assert env["LIGHTNING_RENDER_PORT"] == "1234"


@pytest.mark.parametrize(
    ["value", "expected"],
    (
        ("Yes", True),
        ("yes", True),
        ("YES", True),
        ("Y", True),
        ("y", True),
        ("True", True),
        ("true", True),
        ("TRUE", True),
        ("No", False),
        ("no", False),
        ("NO", False),
        ("N", False),
        ("n", False),
        ("False", False),
        ("false", False),
        ("FALSE", False),
    ),
)
def test_has_panel_autoreload(value, expected):
    """We can get and set autoreload via the environment variable PANEL_AUTORELOAD"""
    with mock.patch.dict(os.environ, {"PANEL_AUTORELOAD": value}):
        assert has_panel_autoreload() == expected
