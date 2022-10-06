import pytest

from lightning_lite.plugins.collectives.collective import Collective


def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        Collective()
