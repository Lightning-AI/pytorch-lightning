import pytest

from lightning.pytorch.utilities.restricted_classmethod import _restricted_classmethod


class RestrictedClass:
    @_restricted_classmethod
    def restricted_cmethod(cls):
        # Can only be called on the class type
        pass

    @classmethod
    def cmethod(cls):
        # Can be called on instance or class type
        pass


def test_restricted_classmethod():
    with pytest.raises(TypeError, match="cannot be called on an instance"):
        RestrictedClass().restricted_cmethod()

    RestrictedClass.restricted_cmethod()  # no exception
