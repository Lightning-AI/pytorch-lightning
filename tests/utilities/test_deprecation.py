import pytest

from pytorch_lightning.utilities.deprecation import deprecated
from tests.helpers.utils import no_warning_call


def my_sum(a=0, b=3):
    return a + b


def my2_sum(a, b):
    return a + b


@deprecated(target=my_sum, ver_deprecate="0.1", ver_remove="0.5")
def dep_sum(a, b=5):
    pass


@deprecated(target=my2_sum, ver_deprecate="0.1", ver_remove="0.5")
def dep2_sum(a, b):
    pass


@deprecated(target=my2_sum, ver_deprecate="0.1", ver_remove="0.5")
def dep3_sum(a, b=4):
    pass


def test_deprecated_func():
    with pytest.deprecated_call(
        match='`tests.utilities.test_deprecation.dep_sum` was deprecated since v0.1 in favor'
        ' of `tests.utilities.test_deprecation.my_sum`. It will be removed in v0.5.'
    ):
        assert dep_sum(2) == 7

    # check that the warning is raised only once per function
    with no_warning_call(DeprecationWarning):
        assert dep_sum(3) == 8

    # and does not affect other functions
    with pytest.deprecated_call(
        match='`tests.utilities.test_deprecation.dep3_sum` was deprecated since v0.1 in favor'
        ' of `tests.utilities.test_deprecation.my2_sum`. It will be removed in v0.5.'
    ):
        assert dep3_sum(2, 1) == 3


def test_deprecated_func_incomplete():

    # missing required argument
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'b'"):
        dep2_sum(2)

    # check that the warning is raised only once per function
    with no_warning_call(DeprecationWarning):
        assert dep2_sum(2, 1) == 3

    # reset the warning
    dep2_sum.warned = False
    # does not affect other functions
    with pytest.deprecated_call(
        match='`tests.utilities.test_deprecation.dep2_sum` was deprecated since v0.1 in favor'
        ' of `tests.utilities.test_deprecation.my2_sum`. It will be removed in v0.5.'
    ):
        assert dep2_sum(b=2, a=1) == 3


class NewCls:

    def __init__(self, c, d="abc"):
        self.my_c = c
        self.my_d = d


class PastCls:

    @deprecated(target=NewCls, ver_deprecate="0.2", ver_remove="0.4")
    def __init__(self, c, d="efg"):
        pass


def test_deprecated_class():
    with pytest.deprecated_call(
        match='`tests.utilities.test_deprecation.PastCls` was deprecated since v0.2 in favor'
        ' of `tests.utilities.test_deprecation.NewCls`. It will be removed in v0.4.'
    ):
        past = PastCls(2)
    assert past.my_c == 2
    assert past.my_d == "efg"

    # check that the warning is raised only once per function
    with no_warning_call(DeprecationWarning):
        assert PastCls(c=2, d="")
