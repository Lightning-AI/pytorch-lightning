import os
from unittest import mock

import pytest
from lightning.app import __package_name__
from lightning.app.utilities.imports import _get_extras, requires


def test_get_extras():
    extras = "app-cloud" if __package_name__ == "lightning" else "cloud"
    extras = _get_extras(extras)
    assert "docker" in extras
    assert "redis" in extras

    assert _get_extras("fake-extras") == ""


@mock.patch.dict(os.environ, {"LIGHTING_TESTING": "0"})
def test_requires():
    @requires("lightning.app")
    def fn():
        pass

    fn()

    @requires("shouldnotexist")
    def fn_raise():
        pass

    with pytest.raises(ModuleNotFoundError, match="Please run: pip install 'shouldnotexist'"):
        fn_raise()

    class ClassRaise:
        @requires("shouldnotexist")
        def __init__(self):
            pass

    with pytest.raises(ModuleNotFoundError, match="Please run: pip install 'shouldnotexist'"):
        ClassRaise()


@mock.patch.dict(os.environ, {"LIGHTING_TESTING": "0"})
def test_requires_multiple():
    @requires(["shouldnotexist1", "shouldnotexist2"])
    def fn_raise():
        pass

    with pytest.raises(ModuleNotFoundError, match="Please run: pip install 'shouldnotexist1' 'shouldnotexist2'"):
        fn_raise()
