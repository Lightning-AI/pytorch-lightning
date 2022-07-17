import pytest

from lightning_app import CloudCompute


def test_cloud_compute_unsupported_features():
    with pytest.raises(ValueError, match="Clusters are't supported yet"):
        CloudCompute("gpu", clusters="as")
    with pytest.raises(ValueError, match="Setting a wait timeout isn't supported yet"):
        CloudCompute("gpu", wait_timeout=1)


def test_cloud_compute_names():
    assert CloudCompute().name == "default"
    assert CloudCompute("cpu-small").name == "cpu-small"
    assert CloudCompute("coconut").name == "coconut"  # the backend is responsible for validation of names


def test_cloud_compute_shared_memory():

    cloud_compute = CloudCompute("gpu", shm_size=1100)
    assert cloud_compute.shm_size == 1100
