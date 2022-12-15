import pytest


def test_removed_gpuaccelerator():
    from pytorch_lightning.accelerators.gpu import GPUAccelerator

    with pytest.raises(NotImplementedError, match="GPUAccelerator`.*no longer supported as of v1.9"):
        GPUAccelerator()

    from pytorch_lightning.accelerators import GPUAccelerator

    with pytest.raises(NotImplementedError, match="GPUAccelerator`.*no longer supported as of v1.9"):
        GPUAccelerator()
