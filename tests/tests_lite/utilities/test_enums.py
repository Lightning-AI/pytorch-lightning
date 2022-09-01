from lightning_lite.utilities import _AcceleratorType
from lightning_lite.utilities.enums import PrecisionType


def test_consistency():
    assert _AcceleratorType.TPU not in ("GPU", "CPU")
    assert _AcceleratorType.TPU in ("TPU", "CPU")
    assert _AcceleratorType.TPU in ("tpu", "CPU")
    assert _AcceleratorType.TPU not in {"GPU", "CPU"}
    # hash cannot be case invariant
    assert _AcceleratorType.TPU not in {"TPU", "CPU"}
    assert _AcceleratorType.TPU in {"tpu", "CPU"}


def test_precision_supported_types():
    assert PrecisionType.supported_types() == ["16", "32", "64", "bf16", "mixed"]
    assert PrecisionType.supported_type(16)
    assert PrecisionType.supported_type("16")
    assert not PrecisionType.supported_type(1)
    assert not PrecisionType.supported_type("invalid")
