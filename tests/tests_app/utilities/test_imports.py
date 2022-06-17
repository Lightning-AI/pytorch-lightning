import os
from unittest import mock

import pytest

from lightning_app.utilities.imports import _module_available, requires


def test_module_available():
    """Test if the 3rd party libs are available."""
    assert _module_available("deepdiff")
    assert _module_available("deepdiff.deephash")
    assert not _module_available("torch.nn.asdf")
    assert not _module_available("asdf")
    assert not _module_available("asdf.bla.asdf")


@mock.patch.dict(os.environ, {"LIGHTING_TESTING": "0"})
def test_requires():
    @requires("lightning_app")
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
