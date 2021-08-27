from pytorch_lightning.utilities import DeviceType
from pytorch_lightning.utilities.enums import PrecisionType


def test_consistency():
    assert DeviceType.TPU not in ("GPU", "CPU")
    assert DeviceType.TPU in ("TPU", "CPU")
    assert DeviceType.TPU in ("tpu", "CPU")
    assert DeviceType.TPU not in {"GPU", "CPU"}
    # hash cannot be case invariant
    assert DeviceType.TPU not in {"TPU", "CPU"}
    assert DeviceType.TPU in {"tpu", "CPU"}


def test_precision_supported_types():
    assert PrecisionType.supported_types() == ["16", "32", "64", "bf16"]
    assert PrecisionType.supported_type(16)
    assert PrecisionType.supported_type("16")
    assert not PrecisionType.supported_type(1)
    assert not PrecisionType.supported_type("invalid")
