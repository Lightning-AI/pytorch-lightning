# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path
from unittest import mock

import fsspec
import torch
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from lightning.fabric.utilities.cloud_io import (
    _atomic_save,
    _checkpoint_join,
    _is_checkpoint_dir,
    _is_dir,
    _prepare_directory_checkpoint,
    _remove_checkpoint,
    _resolve_path,
    get_filesystem,
)


def test_get_filesystem_custom_filesystem():
    _DUMMY_PRFEIX = "dummy"

    class DummyFileSystem(LocalFileSystem): ...

    fsspec.register_implementation(_DUMMY_PRFEIX, DummyFileSystem, clobber=True)
    output_file = os.path.join(f"{_DUMMY_PRFEIX}://", "tmpdir/tmp_file")
    assert isinstance(get_filesystem(output_file), DummyFileSystem)


def test_get_filesystem_local_filesystem():
    assert isinstance(get_filesystem("tmpdir/tmp_file"), LocalFileSystem)


def test_is_dir_with_local_filesystem(tmp_path):
    fs = LocalFileSystem()
    tmp_existing_directory = tmp_path
    tmp_non_existing_directory = tmp_path / "non_existing"

    assert _is_dir(fs, tmp_existing_directory)
    assert not _is_dir(fs, tmp_non_existing_directory)


def test_is_dir_with_object_storage_filesystem():
    class MockAzureBlobFileSystem(AbstractFileSystem):
        def isdir(self, path):
            return path.startswith("azure://") and not path.endswith(".txt")

        def isfile(self, path):
            return path.startswith("azure://") and path.endswith(".txt")

    class MockGCSFileSystem(AbstractFileSystem):
        def isdir(self, path):
            return path.startswith("gcs://") and not path.endswith(".txt")

        def isfile(self, path):
            return path.startswith("gcs://") and path.endswith(".txt")

    class MockS3FileSystem(AbstractFileSystem):
        def isdir(self, path):
            return path.startswith("s3://") and not path.endswith(".txt")

        def isfile(self, path):
            return path.startswith("s3://") and path.endswith(".txt")

    fsspec.register_implementation("azure", MockAzureBlobFileSystem, clobber=True)
    fsspec.register_implementation("gcs", MockGCSFileSystem, clobber=True)
    fsspec.register_implementation("s3", MockS3FileSystem, clobber=True)

    azure_directory = "azure://container/directory/"
    azure_file = "azure://container/file.txt"
    gcs_directory = "gcs://bucket/directory/"
    gcs_file = "gcs://bucket/file.txt"
    s3_directory = "s3://bucket/directory/"
    s3_file = "s3://bucket/file.txt"

    assert _is_dir(get_filesystem(azure_directory), azure_directory)
    assert _is_dir(get_filesystem(azure_directory), azure_directory, strict=True)
    assert not _is_dir(get_filesystem(azure_directory), azure_file)
    assert not _is_dir(get_filesystem(azure_directory), azure_file, strict=True)

    assert _is_dir(get_filesystem(gcs_directory), gcs_directory)
    assert _is_dir(get_filesystem(gcs_directory), gcs_directory, strict=True)
    assert not _is_dir(get_filesystem(gcs_directory), gcs_file)
    assert not _is_dir(get_filesystem(gcs_directory), gcs_file, strict=True)

    assert _is_dir(get_filesystem(s3_directory), s3_directory)
    assert _is_dir(get_filesystem(s3_directory), s3_directory, strict=True)
    assert not _is_dir(get_filesystem(s3_directory), s3_file)
    assert not _is_dir(get_filesystem(s3_directory), s3_file, strict=True)


def test_atomic_save_uses_pipe_for_s3(tmp_path):
    """Test that _atomic_save uses fs.pipe() for S3 filesystems."""
    checkpoint = {"key": torch.tensor([1, 2, 3])}
    filepath = "s3://bucket/checkpoint.ckpt"

    mock_fs = mock.MagicMock()
    mock_fs.__class__.__name__ = "S3FileSystem"

    with (
        mock.patch("lightning.fabric.utilities.cloud_io._is_object_storage", return_value=True),
        mock.patch("fsspec.core.url_to_fs", return_value=(mock_fs, "bucket/checkpoint.ckpt")),
    ):
        _atomic_save(checkpoint, filepath)

    mock_fs.pipe.assert_called_once()
    mock_fs.open.assert_not_called()


def test_atomic_save_uses_write_for_azure(tmp_path):
    """Test that _atomic_save uses f.write() for Azure filesystems."""
    import sys
    import types

    checkpoint = {"key": torch.tensor([1, 2, 3])}
    filepath = "azure://container/checkpoint.ckpt"

    # Create a fake adlfs module so isinstance check works
    AzureBlobFileSystem = type("AzureBlobFileSystem", (), {})
    fake_adlfs = types.ModuleType("adlfs")
    fake_adlfs.AzureBlobFileSystem = AzureBlobFileSystem

    mock_fs = mock.MagicMock()
    mock_fs.__class__ = AzureBlobFileSystem

    with (
        mock.patch.dict(sys.modules, {"adlfs": fake_adlfs}),
        mock.patch("lightning.fabric.utilities.cloud_io.module_available", return_value=True),
        mock.patch("lightning.fabric.utilities.cloud_io._is_object_storage", return_value=True),
        mock.patch("fsspec.core.url_to_fs", return_value=(mock_fs, "container/checkpoint.ckpt")),
    ):
        _atomic_save(checkpoint, filepath)

    mock_fs.pipe.assert_not_called()
    mock_fs.open.assert_called_once()


def test_atomic_save_uses_write_for_local(tmp_path):
    """Test that _atomic_save uses f.write() for local filesystems."""
    checkpoint = {"key": torch.tensor([1, 2, 3])}
    filepath = tmp_path / "checkpoint.ckpt"

    _atomic_save(checkpoint, filepath)

    assert filepath.exists()
    loaded = torch.load(filepath, weights_only=True)
    torch.testing.assert_close(loaded["key"], checkpoint["key"])


def test_resolve_path_local_vs_remote(tmp_path):
    resolved = _resolve_path(str(tmp_path / "ckpt"))
    assert isinstance(resolved, Path)
    assert resolved == tmp_path / "ckpt"

    # build the URI from tmp_path: a hardcoded "file:///tmp/..." is not absolute on Windows,
    # where fsspec would prepend the current drive (e.g. "D:/tmp/...")
    local_file = tmp_path / "test.txt"
    resolved_file_uri = _resolve_path(local_file.as_uri())
    assert isinstance(resolved_file_uri, Path)
    assert resolved_file_uri == local_file

    # a hand-written drive-less URI: on Windows fsspec resolves it against the current
    # drive, matching os.path.abspath semantics; on POSIX it is returned unchanged
    resolved_literal = _resolve_path("file:///tmp/test.txt")
    assert isinstance(resolved_literal, Path)
    assert resolved_literal == Path(os.path.abspath("/tmp/test.txt"))

    resolved = _resolve_path("gs://bucket/checkpoints/epoch=1.ckpt")
    assert resolved == "gs://bucket/checkpoints/epoch=1.ckpt"
    assert isinstance(resolved, str)


def test_checkpoint_join_does_not_corrupt_remote_urls():
    assert _checkpoint_join(Path("/tmp/ckpt"), "meta.pt") == Path("/tmp/ckpt/meta.pt")
    assert _checkpoint_join("gs://bucket/ckpt", "meta.pt") == "gs://bucket/ckpt/meta.pt"
    assert _checkpoint_join("gs://bucket/ckpt/", "meta.pt") == "gs://bucket/ckpt/meta.pt"


def test_is_checkpoint_dir_local(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    f = tmp_path / "a_file"
    f.write_text("x")
    assert _is_checkpoint_dir(d) is True
    assert _is_checkpoint_dir(f) is False
    assert _is_checkpoint_dir(tmp_path / "missing") is False


def test_is_checkpoint_dir_remote():
    fs = fsspec.filesystem("memory")
    fs.mkdir("/r/adir")
    with fs.open("/r/a_file", "wb") as f:
        f.write(b"x")
    assert _is_checkpoint_dir("memory:///r/adir") is True
    assert _is_checkpoint_dir("memory:///r/a_file") is False
    assert _is_checkpoint_dir("memory:///r/missing") is False


def test_prepare_directory_checkpoint_local_replaces_file(tmp_path):
    p = tmp_path / "ckpt"
    p.write_text("stray file")
    _prepare_directory_checkpoint(p)
    assert p.is_dir()


def test_prepare_directory_checkpoint_remote_memory():
    fs = fsspec.filesystem("memory")
    with fs.open("/m/ckpt", "wb") as f:
        f.write(b"stray")
    _prepare_directory_checkpoint("memory:///m/ckpt")
    assert not fs.isfile("/m/ckpt")
    assert fs.isdir("/m/ckpt")


def test_remove_checkpoint_local_file_and_dir(tmp_path):
    f = tmp_path / "f.ckpt"
    f.write_text("x")
    _remove_checkpoint(f)
    assert not f.exists()
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "x").write_text("y")
    _remove_checkpoint(d)
    assert not d.exists()


def test_remove_checkpoint_remote_memory():
    fs = fsspec.filesystem("memory")
    with fs.open("/r/ckpt/x", "wb") as f:
        f.write(b"a")
    _remove_checkpoint("memory:///r/ckpt")
    assert not fs.exists("/r/ckpt")


def test_remove_checkpoint_remote_file_memory():
    fs = fsspec.filesystem("memory")
    with fs.open("/r/file.ckpt", "wb") as f:
        f.write(b"a")
    _remove_checkpoint("memory:///r/file.ckpt")
    assert not fs.exists("/r/file.ckpt")
