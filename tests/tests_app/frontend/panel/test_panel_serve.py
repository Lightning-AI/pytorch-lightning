"""The panel_serve_render_fn_or_file file gets run by Python to lunch a Panel Server with
Lightning."""
import os
from unittest import mock

import pytest

from lightning_app.frontend.panel.panel_serve_render_fn_or_file import _has_autoreload


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
def test_autoreload(value, expected):
    """We can get and set autoreload via the environment variable PANEL_AUTORELOAD"""
    with mock.patch.dict(os.environ, {"PANEL_AUTORELOAD": value}):
        assert _has_autoreload() == expected
