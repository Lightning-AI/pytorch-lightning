from pytorch_lightning.utilities import DeviceType


def test_consistency():
    assert DeviceType.TPU not in ("GPU", "CPU")
    assert DeviceType.TPU in ("TPU", "CPU")
    assert DeviceType.TPU in ("tpu", "CPU")
    assert DeviceType.TPU not in {"GPU", "CPU"}
    # hash cannot be case invariant
    assert DeviceType.TPU not in {"TPU", "CPU"}
    assert DeviceType.TPU in {"tpu", "CPU"}
