import pytest


def test_v2_0_0_unsupported_base_loop():
    from pytorch_lightning.loops.base import Loop

    with pytest.raises(NotImplementedError, match="Loop` was deprecated in v1.7.0 and removed as of v1.9"):
        Loop()
