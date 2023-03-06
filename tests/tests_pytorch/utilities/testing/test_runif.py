import pytest

from lightning.pytorch.utilities.testing import _RunIf


@_RunIf(min_torch="99")
def test_always_skip():
    exit(1)


@pytest.mark.parametrize("arg1", [0.5, 1.0, 2.0])
@_RunIf(min_torch="0.0")
def test_wrapper(arg1: float):
    assert arg1 > 0.0
