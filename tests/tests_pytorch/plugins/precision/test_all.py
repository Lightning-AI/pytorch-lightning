import pytest
import torch
from lightning.pytorch.plugins import (
    DeepSpeedPrecisionPlugin,
    DoublePrecisionPlugin,
    FSDPPrecisionPlugin,
    HalfPrecisionPlugin,
)


@pytest.mark.parametrize(
    "precision",
    [
        DeepSpeedPrecisionPlugin("16-true"),
        DoublePrecisionPlugin(),
        HalfPrecisionPlugin(),
        "fsdp",
    ],
)
def test_default_dtype_is_restored(precision):
    if precision == "fsdp":
        precision = FSDPPrecisionPlugin("16-true")

    contexts = (
        (precision.module_init_context, precision.forward_context)
        if not isinstance(precision, DeepSpeedPrecisionPlugin)
        else (precision.module_init_context,)
    )
    for context in contexts:
        assert torch.get_default_dtype() is torch.float32
        with pytest.raises(RuntimeError, match="foo"), context():
            assert torch.get_default_dtype() is not torch.float32
            raise RuntimeError("foo")
        assert torch.get_default_dtype() is torch.float32
