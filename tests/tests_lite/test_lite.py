from lightning_lite.lite import LightningLite


def test_placeholder(tmpdir):
    assert True


@RunIf(min_cuda_gpus=2, standalone=True)
def test_placeholder_standalone(tmpdir):
    assert True
