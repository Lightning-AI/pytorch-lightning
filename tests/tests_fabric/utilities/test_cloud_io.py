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
import tempfile
from pathlib import Path
from unittest import mock

import fsspec
import pytest
import torch
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from lightning.fabric.utilities.cloud_io import _atomic_save, _is_dir, get_filesystem


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


def test_atomic_save_local_stages_next_to_destination(tmp_path, monkeypatch):
    """Regression for #21253: local checkpoints must stage in the destination dir, not $TMPDIR.

    Setups with a small $TMPDIR (SLURM, HPC clusters) hit "no space left on device" when a
    large checkpoint is staged there. Stage next to the destination instead. Detected by
    instrumenting ``tempfile.mkstemp`` — the staging directory is observable mid-write, not
    after, because a successful transaction cleans the staging file up on commit. Paths are
    resolved before comparison so this doesn't false-fail on macOS where /tmp is a symlink to
    /private/tmp.

    """
    sentinel_tmpdir = (tmp_path / "sentinel_tmpdir").resolve()
    dest_dir = (tmp_path / "destination").resolve()
    sentinel_tmpdir.mkdir()
    dest_dir.mkdir()

    monkeypatch.setenv("TMPDIR", str(sentinel_tmpdir))
    monkeypatch.setattr(tempfile, "tempdir", str(sentinel_tmpdir))

    real_mkstemp = tempfile.mkstemp
    staged_parents = []

    def traced_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        staged_parents.append(Path(name).resolve().parent)
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", traced_mkstemp)

    filepath = dest_dir / "checkpoint.ckpt"
    _atomic_save({"key": torch.tensor([1, 2, 3])}, filepath)

    assert filepath.exists()
    assert sentinel_tmpdir not in staged_parents, f"mkstemp staged through $TMPDIR: {staged_parents}"
    assert dest_dir in staged_parents, f"mkstemp did not stage in destination dir: {staged_parents}"


def test_atomic_save_local_cleans_up_staging_on_failure(tmp_path):
    """If the rename fails, the staging file must not leak in the destination dir.

    Patches os.replace to fail so we can observe what happens after a successful write but a failed rename — the
    destination should not exist and the destination dir should be empty (i.e. no staging file under any naming
    convention).

    """
    filepath = tmp_path / "checkpoint.ckpt"

    with mock.patch("lightning.fabric.utilities.cloud_io.os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="boom"):
            _atomic_save({"key": torch.tensor([1, 2, 3])}, filepath)

    assert not filepath.exists()
    assert list(tmp_path.iterdir()) == [], f"unexpected files left after failed save: {list(tmp_path.iterdir())}"


def test_atomic_save_local_preserves_existing_on_failure(tmp_path):
    """The atomicity guarantee: a failed save must not corrupt or destroy a prior checkpoint.

    The most important property of _atomic_save. Writes a baseline checkpoint, then attempts a
    save that fails at the rename step, then asserts the original bytes are still on disk
    untouched.
    """
    filepath = tmp_path / "checkpoint.ckpt"
    _atomic_save({"key": torch.tensor([0, 0, 0])}, filepath)
    original_bytes = filepath.read_bytes()

    with mock.patch("lightning.fabric.utilities.cloud_io.os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="boom"):
            _atomic_save({"key": torch.tensor([42, 42, 42])}, filepath)

    assert filepath.read_bytes() == original_bytes
    assert [p for p in tmp_path.iterdir() if p != filepath] == []


def test_atomic_save_local_missing_parent_raises(tmp_path):
    """Parent directories are not auto-created — locks in current behavior.

    Lightning's checkpoint code creates dirs upstream (ModelCheckpoint.setup); a future refactor silently creating them
    here would mask caller bugs.

    """
    filepath = tmp_path / "missing" / "checkpoint.ckpt"

    with pytest.raises(FileNotFoundError):
        _atomic_save({"key": torch.tensor([1, 2, 3])}, filepath)

    assert not filepath.exists()
    assert not filepath.parent.exists()
