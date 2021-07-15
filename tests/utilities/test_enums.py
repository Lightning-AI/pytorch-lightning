from pytorch_lightning.utilities import DeviceType


def test_consistency():
    assert DeviceType.TPU in ('tpu', 'CPU')
    assert DeviceType.TPU in ('tpu', 'CPU')
