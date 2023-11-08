import pytest
from lightning.app.storage.mount import Mount


def test_create_s3_mount_successfully():
    mount = Mount(source="s3://foo/bar/", mount_path="/foo")
    assert mount.source == "s3://foo/bar/"
    assert mount.mount_path == "/foo"
    assert mount.protocol == "s3://"


def test_create_non_s3_mount_fails():
    with pytest.raises(ValueError, match="Unknown protocol for the mount 'source' argument"):
        Mount(source="foo/bar/", mount_path="/foo")

    with pytest.raises(ValueError, match="Unknown protocol for the mount 'source' argument"):
        Mount(source="gcs://foo/bar/", mount_path="/foo")

    with pytest.raises(ValueError, match="Unknown protocol for the mount 'source' argument"):
        Mount(source="3://foo/bar/", mount_path="/foo")


def test_create_s3_mount_without_directory_prefix_fails():
    with pytest.raises(ValueError, match="S3 mounts must end in a trailing slash"):
        Mount(source="s3://foo/bar", mount_path="/foo")

    with pytest.raises(ValueError, match="S3 mounts must end in a trailing slash"):
        Mount(source="s3://foo", mount_path="/foo")


def test_create_mount_without_mount_path_argument():
    m = Mount(source="s3://foo/")
    assert m.mount_path == "/data/foo"

    m = Mount(source="s3://foo/bar/")
    assert m.mount_path == "/data/bar"


def test_create_mount_path_with_relative_path_errors():
    with pytest.raises(ValueError, match="mount_path argument must be an absolute path"):
        Mount(source="s3://foo/", mount_path="./doesnotwork")
