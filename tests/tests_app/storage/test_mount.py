import pytest

from lightning_app.storage.mount import Mount


def test_create_s3_mount_successfully():
    mount = Mount(source="s3://foo/bar/", root_dir="./foo")
    assert mount.source == "s3://foo/bar/"
    assert mount.root_dir == "./foo"
    assert mount.protocol == "s3://"


def test_create_non_s3_mount_fails():
    with pytest.raises(ValueError, match="Unknown protocol for the mount 'source' argument"):
        Mount(source="foo/bar/", root_dir="./foo")

    with pytest.raises(ValueError, match="Unknown protocol for the mount 'source' argument"):
        Mount(source="gcs://foo/bar/", root_dir="./foo")

    with pytest.raises(ValueError, match="Unknown protocol for the mount 'source' argument"):
        Mount(source="3://foo/bar/", root_dir="./foo")


def test_create_s3_mount_without_directory_prefix_fails():
    with pytest.raises(ValueError, match="S3 mounts must end in a trailing slash"):
        Mount(source="s3://foo/bar", root_dir="./foo")

    with pytest.raises(ValueError, match="S3 mounts must end in a trailing slash"):
        Mount(source="s3://foo", root_dir="./foo")


def test_create_mount_without_root_dir_argument_fails():
    with pytest.raises(ValueError, match="The mount for `source` `s3://foo/` does not set the required `root_dir`"):
        Mount(source="s3://foo/", root_dir="")
