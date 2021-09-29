from pytorch_lightning.utilities.enums import DeviceType, GradClipAlgorithmType


def test_consistency():
    assert DeviceType.TPU not in ("GPU", "CPU")
    assert DeviceType.TPU in ("TPU", "CPU")
    assert DeviceType.TPU in ("tpu", "CPU")
    assert DeviceType.TPU not in {"GPU", "CPU"}
    # hash cannot be case invariant
    assert DeviceType.TPU not in {"TPU", "CPU"}
    assert DeviceType.TPU in {"tpu", "CPU"}


def test_gradient_clip_algorithms():
    assert GradClipAlgorithmType.supported_types() == ["value", "norm"]
    assert GradClipAlgorithmType.supported_type("norm")
    assert GradClipAlgorithmType.supported_type("value")
    assert not GradClipAlgorithmType.supported_type("norm2")
