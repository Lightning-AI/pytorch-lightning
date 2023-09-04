import pytest

from lightning.pytorch.utilities.restricted_classmethod import restricted_classmethod


class _C:
    @restricted_classmethod
    def _rcmethod(cls):
        pass


def test_restricted_classmethod():
    c = _C()

    with pytest.raises(TypeError):
        c._rcmethod()

    _C._rcmethod()  # no exception
