import pytest

from pl_sandbox.my_module import my_sample_func


@pytest.mark.parametrize(
    "a,b,expected",
    [
        pytest.param(1, 2, 3),
        pytest.param(-1, 1.0, 0),
    ],
)
def test_sample_func(a, b, expected):
    """Sample test case with parametrization."""
    assert my_sample_func(a, b) == expected
