"""Testing `exceptions.py`"""

import pytest

from lightning_lite.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.exceptions import _ValueError
from tests_pytorch.deprecated_api import no_deprecated_call


def test_exception_deprecations():
    with no_deprecated_call():
        x = _ValueError("test")
        assert isinstance(x, _ValueError)  # this needs to work for proper instantiation
    with pytest.deprecated_call(match="Please check with `ValueError"):
        assert isinstance(x, MisconfigurationException)
    with no_deprecated_call():
        assert isinstance(x, ValueError)

    with pytest.deprecated_call(match="Using `MisconfigurationException` is deprecated"):
        y = MisconfigurationException("this is deprecated")
    with no_deprecated_call():
        assert isinstance(y, MisconfigurationException)
    with no_deprecated_call():
        assert not isinstance(y, ValueError)

    try:
        raise x
    except MisconfigurationException:
        pass

    try:
        raise x
    except ValueError:
        pass
