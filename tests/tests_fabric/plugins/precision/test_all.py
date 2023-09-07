import pytest
import torch

from lightning.fabric.plugins import DeepSpeedPrecision, DoublePrecision, FSDPPrecision, HalfPrecision
from tests_fabric.helpers.runif import RunIf


@pytest.mark.parametrize(
    "precision",
    [
        DeepSpeedPrecision("16-true"),
        DoublePrecision(),
        HalfPrecision(),
        pytest.param(FSDPPrecision("16-true"), marks=RunIf(min_torch="1.12")),
    ],
)
def test_default_dtype_is_restored(precision):
    assert torch.get_default_dtype() is torch.float32
    with pytest.raises(RuntimeError, match="foo"), precision.init_context():
        assert torch.get_default_dtype() is not torch.float32
        raise RuntimeError("foo")
    assert torch.get_default_dtype() is torch.float32
