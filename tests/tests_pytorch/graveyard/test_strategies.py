import pytest


def test_v2_0_0_lightningdeepspeedmodule():
    from pytorch_lightning.strategies.deepspeed import LightningDeepSpeedModule

    with pytest.raises(RuntimeError, match="LightningDeepSpeedModule` was deprecated in v1.7.1"):
        LightningDeepSpeedModule()
