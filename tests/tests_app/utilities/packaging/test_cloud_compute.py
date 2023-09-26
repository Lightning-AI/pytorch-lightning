import pytest
from lightning.app import CloudCompute
from lightning.app.storage import Mount


def test_cloud_compute_names():
    assert CloudCompute().name == "cpu-small"
    assert CloudCompute("cpu-small").name == "cpu-small"
    assert CloudCompute("coconut").name == "coconut"  # the backend is responsible for validation of names


def test_cloud_compute_shared_memory():
    cloud_compute = CloudCompute("gpu", shm_size=1100)
    assert cloud_compute.shm_size == 1100

    cloud_compute = CloudCompute("gpu")
    assert cloud_compute.shm_size == 1024

    cloud_compute = CloudCompute("cpu")
    assert cloud_compute.shm_size == 0


def test_cloud_compute_with_mounts():
    mount_1 = Mount(source="s3://foo/", mount_path="/foo")
    mount_2 = Mount(source="s3://foo/bar/", mount_path="/bar")

    cloud_compute = CloudCompute("gpu", mounts=mount_1)
    assert cloud_compute.mounts == mount_1

    cloud_compute = CloudCompute("gpu", mounts=[mount_1, mount_2])
    assert cloud_compute.mounts == [mount_1, mount_2]

    cc_dict = cloud_compute.to_dict()
    assert "mounts" in cc_dict
    assert cc_dict["mounts"] == [
        {"mount_path": "/foo", "source": "s3://foo/"},
        {"mount_path": "/bar", "source": "s3://foo/bar/"},
    ]

    assert CloudCompute.from_dict(cc_dict) == cloud_compute


def test_cloud_compute_with_non_unique_mount_root_dirs():
    mount_1 = Mount(source="s3://foo/", mount_path="/foo")
    mount_2 = Mount(source="s3://foo/bar/", mount_path="/foo")

    with pytest.raises(ValueError, match="Every Mount attached to a work must have a unique"):
        CloudCompute("gpu", mounts=[mount_1, mount_2])


def test_cloud_compute_clone():
    c1 = CloudCompute("gpu")
    c2 = c1.clone()

    assert isinstance(c2, CloudCompute)

    c1_dict = c1.to_dict()
    c2_dict = c2.to_dict()

    assert len(c1_dict) == len(c2_dict)

    for k in c1_dict:
        if k == "_internal_id":
            assert c1_dict[k] != c2_dict[k]
        else:
            assert c1_dict[k] == c2_dict[k]


def test_interruptible(monkeypatch):
    """Test interruptible can be enabled with env variables and for GPU only."""
    with pytest.raises(ValueError, match="isn't supported yet"):
        CloudCompute("gpu", interruptible=True)

    monkeypatch.setenv("LIGHTNING_INTERRUPTIBLE_WORKS", "1")
    with pytest.raises(ValueError, match="supported only with GPU"):
        CloudCompute("cpu", interruptible=True)

    cloud_compute = CloudCompute("gpu", interruptible=True)
    assert hasattr(cloud_compute, "interruptible")
    # TODO: To be removed once the platform is updated.
    assert hasattr(cloud_compute, "preemptible")
