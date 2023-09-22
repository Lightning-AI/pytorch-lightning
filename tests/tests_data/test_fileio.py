import os
from unittest import mock

import pytest
from lightning.data.fileio import OpenCloudFileObj, is_path, is_url, open_single_file, path_to_url


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("s3://my_bucket/a", True),
        ("s3:/my_bucket", False),
        ("my_bucket", False),
        ("my_bucket_s3://", False),
    ],
)
def test_is_url(input_str, expected):
    assert is_url(input_str) == expected


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("s3://my_bucket/a", False),
        ("s3:/my_bucket", False),
        ("my_bucket", False),
        ("my_bucket_s3://", False),
        ("/my_bucket", True),
    ],
)
def test_is_path(input_str, expected):
    assert is_path(input_str) == expected


@pytest.mark.parametrize(
    ("path", "bucket_name", "bucket_root_path", "expected"),
    [
        ("/data/abc/def", "my_bucket", "/data/abc", "s3://my_bucket/def"),
        ("/data/abc/def", "my_bucket", "/data", "s3://my_bucket/abc/def"),
    ],
)
def test_path_to_url(path, bucket_name, bucket_root_path, expected):
    assert path_to_url(path, bucket_name, bucket_root_path) == expected


def test_path_to_url_error():
    with pytest.raises(ValueError, match="Cannot create a path from /path1/abc relative to /path2"):
        path_to_url("/path1/abc", "foo", "/path2")


@pytest.mark.parametrize("path", ["s3://my_bucket/da.txt", "abc.txt"])
@mock.patch("s3fs.S3FileSystem", autospec=True)
def test_read_single_file_read(patch: mock.Mock, path, tmp_path):
    from torchdata.datapipes.utils import StreamWrapper

    is_s3 = is_url(path)

    if not is_s3:
        path = os.path.join(tmp_path, path)
        with open(path, "w") as f:
            f.write("mytestfile")

    file_stream = open_single_file(path)
    assert isinstance(file_stream, StreamWrapper)

    content = file_stream.read()

    if is_s3:
        assert isinstance(file_stream.file_obj, mock.Mock)
        assert patch.open.assert_called_once

    else:
        assert content == "mytestfile"


@pytest.mark.parametrize("path", ["s3://my_bucket/da.txt", "abc.txt"])
@mock.patch("s3fs.S3FileSystem", autospec=True)
def test_read_single_file_write(patch: mock.Mock, path, tmp_path):
    from torchdata.datapipes.utils import StreamWrapper

    is_s3 = is_url(path)

    if not is_s3:
        path = os.path.join(tmp_path, path)

    file_stream = open_single_file(path, mode="w")
    assert isinstance(file_stream, StreamWrapper)
    file_stream.write("mytestfile")
    file_stream.close()

    if is_s3:
        assert isinstance(file_stream.file_obj, mock.Mock)
        assert patch.open.assert_called_once

    else:
        with open(path) as f:
            assert f.read() == "mytestfile"


def test_open_cloud_file_obj(tmp_path):
    path = os.path.join(tmp_path, "foo.txt")
    with open(path, "w") as f:
        f.write("bar!")

    f = OpenCloudFileObj(path)

    with f:
        assert f.read() == "bar!"
    assert f._stream.closed

    f = OpenCloudFileObj(path)
    assert f.read() == "bar!"
    f.close()
    assert f._stream.closed

    with OpenCloudFileObj(path, "w") as f:
        f.write("not bar anymore!")

    with open(path) as f:
        assert f.read() == "not bar anymore!"
