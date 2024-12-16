import warnings
from re import escape

import pytest

from lightning_utilities.test.warning import no_warning_call


def test_no_warning_call():
    with no_warning_call():
        ...

    with pytest.raises(AssertionError, match=escape("`Warning` was raised: UserWarning('foo')")), no_warning_call():
        warnings.warn("foo")

    with no_warning_call(DeprecationWarning):
        warnings.warn("foo")

    class MyDeprecationWarning(DeprecationWarning): ...

    with (
        pytest.raises(AssertionError, match=escape("`DeprecationWarning` was raised: MyDeprecationWarning('bar')")),
        no_warning_call(DeprecationWarning),
    ):
        warnings.warn("bar", category=MyDeprecationWarning)
