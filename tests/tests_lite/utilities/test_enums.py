from lightning_lite.utilities.enums import PrecisionType


def test_precision_supported_types():
    assert PrecisionType.supported_types() == ["16", "32", "64", "bf16", "mixed"]
    assert PrecisionType.supported_type(16)
    assert PrecisionType.supported_type("16")
    assert not PrecisionType.supported_type(1)
    assert not PrecisionType.supported_type("invalid")
