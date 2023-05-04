import pytest

from lightning.pytorch.strategies import DDPStrategy


def test_ddp_is_distributed():
    strategy = DDPStrategy()
    with pytest.deprecated_call(match="is deprecated"):
        _ = strategy.is_distributed
