import pytest

from pytorch_lightning.utilities.deprecation import deprecated


def my_sum(a, b=3):
    return a + b


@deprecated(target=my_sum, ver_deprecate="0.1", ver_remove="0.5")
def dep_sum(a, b):
    pass


@deprecated(target=my_sum, ver_deprecate="0.1", ver_remove="0.5")
def dep2_sum(a, b):
    pass


def test_deprecated_func():
    with pytest.deprecated_call(
        match='This `dep_sum` was deprecated since v0.1 in favor of `tests.utilities.test_deprecation.my_sum`.'
        ' It will be removed in v0.5.'
    ):
        assert dep_sum(2, b=5) == 7

    # check that the warning is raised only once per function
    with pytest.warns(None) as record:
        assert dep_sum(2, b=5) == 7
    assert len(record) == 0

    # and does not affect other functions
    with pytest.deprecated_call(
        match='This `dep2_sum` was deprecated since v0.1 in favor of `tests.utilities.test_deprecation.my_sum`.'
        ' It will be removed in v0.5.'
    ):
        assert dep2_sum(2) == 5
