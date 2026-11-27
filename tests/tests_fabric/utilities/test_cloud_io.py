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
import errno
import glob
import hashlib
import io
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fsspec
import pytest
import torch
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from lightning.fabric.utilities import cloud_io
from lightning.fabric.utilities.cloud_io import (
    _atomic_save,
    _checkpoint_join,
    _is_checkpoint_dir,
    _is_dir,
    _load,
    _prepare_directory_checkpoint,
    _remote_version,
    _remove_checkpoint,
    _resolve_path,
    _user_cache_prefix,
    clear_cache,
    get_filesystem,
)

_requires_cache = pytest.mark.skipif(not cloud_io._HAS_FCNTL, reason="the node-local checkpoint cache requires `fcntl`")


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


def test_atomic_save_permission_error_propagation():
    """Test that _atomic_save propagates PermissionError if it is not an EXDEV error."""
    checkpoint = {"key": torch.tensor([1, 2, 3])}
    filepath = "memory://checkpoint.ckpt"

    mock_fs = mock.MagicMock()
    mock_fs.open.side_effect = PermissionError("Permission denied")
    mock_fs.pipe.side_effect = PermissionError("Permission denied")

    with (
        mock.patch("fsspec.core.url_to_fs", return_value=(mock_fs, "checkpoint.ckpt")),
        pytest.raises(PermissionError, match="Permission denied"),
    ):
        _atomic_save(checkpoint, filepath)


def test_atomic_save_exdev_to_runtime_error():
    """Test that _atomic_save maps PermissionError with EXDEV context to RuntimeError."""
    checkpoint = {"key": torch.tensor([1, 2, 3])}
    filepath = "memory://checkpoint.ckpt"

    mock_fs = mock.MagicMock()

    os_error = OSError(errno.EXDEV, "Cross-device link")
    permission_error = PermissionError("Permission denied")
    permission_error.__context__ = os_error

    mock_fs.open.side_effect = permission_error
    mock_fs.pipe.side_effect = permission_error

    with (
        mock.patch("fsspec.core.url_to_fs", return_value=(mock_fs, "checkpoint.ckpt")),
        pytest.raises(RuntimeError, match="Upgrade fsspec to enable cross-device local checkpoints"),
    ):
        _atomic_save(checkpoint, filepath)


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
    assert resolved_literal == Path(os.path.abspath("/tmp/test.txt")).resolve()

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


def test_atomic_save_streams_to_local_file_without_buffering(tmp_path):
    """Local saves stream straight to the file handle instead of holding a second full copy in memory."""
    filepath = tmp_path / "checkpoint.ckpt"
    saved_targets = []
    real_torch_save = torch.save

    def spy_save(obj, f, *args, **kwargs):
        saved_targets.append(f)
        return real_torch_save(obj, f, *args, **kwargs)

    with mock.patch("lightning.fabric.utilities.cloud_io.torch.save", side_effect=spy_save):
        _atomic_save({"key": torch.tensor([1, 2, 3])}, filepath)

    assert filepath.exists()
    # serialized exactly once, directly into the file handle rather than an in-memory BytesIO copy
    assert len(saved_targets) == 1
    assert not isinstance(saved_targets[0], io.BytesIO)
    loaded = torch.load(filepath, weights_only=True)
    torch.testing.assert_close(loaded["key"], torch.tensor([1, 2, 3]))


class _RaisesOnPickle:
    """Object that makes torch.save fail partway through serialization."""

    def __reduce__(self):
        raise RuntimeError("simulated crash inside torch.save")


def test_atomic_save_local_interrupted_save_creates_no_partial_file(tmp_path):
    """A torch.save that raises mid-serialization must not leave a partial file at a new path."""
    filepath = tmp_path / "checkpoint.ckpt"
    checkpoint = {"weights": torch.zeros(1_000_000), "poison": _RaisesOnPickle()}

    with pytest.raises(RuntimeError, match="simulated crash"):
        _atomic_save(checkpoint, filepath)

    assert not filepath.exists()
    assert os.listdir(tmp_path) == []


def _big_checkpoint(path, fill=0.0):
    """Write a checkpoint above the monkeypatched _CACHE_MIN_SIZE_BYTES threshold (1024 bytes)."""
    torch.save({"weights": torch.full((4096,), fill, dtype=torch.float32)}, path)
    return os.path.getsize(path)


def _versioned_fs(src, size, version="v1", calls=None):
    """A stub filesystem that reports a version token, like S3/GCS do."""

    class DummyFS:
        def info(self, path):
            return {"size": size, "etag": version}

        def open(self, path, mode):
            return open(src, mode)

        def get_file(self, rpath, lpath):
            if calls is not None:
                calls.append((rpath, lpath))
            shutil.copyfile(src, lpath)

    return DummyFS()


def _use_tmp_cache_root(tmp_path, monkeypatch):
    """Route all cache roots into `tmp_path` and lower the size threshold for fast unit tests."""
    if not cloud_io._HAS_FCNTL:  # pragma: no cover
        pytest.skip("the node-local checkpoint cache requires `fcntl`")
    for var in ("LIGHTNING_CHECKPOINT_CACHE", "LIGHTNING_CHECKPOINT_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._CACHE_MIN_SIZE_BYTES", 1024)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._is_local_file_protocol", lambda _: False)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._get_cache_roots", lambda: (str(tmp_path),))


def test_load_local_file_uri(tmp_path):
    """file:// URIs must be stripped before calling torch.load."""
    checkpoint = {"weights": torch.tensor([1.0, 2.0, 3.0])}
    ckpt_path = tmp_path / "local_uri.ckpt"
    torch.save(checkpoint, ckpt_path)

    file_uri = ckpt_path.as_uri()
    assert file_uri.startswith("file://")
    loaded = _load(file_uri, map_location="cpu")
    torch.testing.assert_close(loaded["weights"], checkpoint["weights"])


def test_load_remote_small_file_streaming(tmp_path, monkeypatch):
    checkpoint = {"weights": torch.tensor([1.0, 2.0, 3.0])}
    ckpt_path = tmp_path / "small.ckpt"
    torch.save(checkpoint, ckpt_path)

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._is_local_file_protocol", lambda _: False)
    loaded = _load(str(ckpt_path), map_location="cpu")
    torch.testing.assert_close(loaded["weights"], checkpoint["weights"])


def test_load_remote_size_none_and_version_zero(tmp_path, monkeypatch):
    """fs.info() returning size=None must stream without TypeError; generation=0 must be accepted as a valid
    version."""
    ckpt_path = tmp_path / "size_none.ckpt"
    size = _big_checkpoint(ckpt_path, fill=3.0)
    _use_tmp_cache_root(tmp_path, monkeypatch)

    class SizeNoneFS:
        def info(self, path):
            return {"size": None, "etag": "v1"}

        def open(self, path, mode):
            return open(ckpt_path, mode)

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: SizeNoneFS())
    loaded = _load(str(ckpt_path), map_location="cpu")
    assert torch.all(loaded["weights"] == 3.0)

    # Now test generation=0 (numeric zero version token should cache, not be treated as falsy)
    calls = []

    class GenZeroFS:
        def info(self, path):
            return {"size": size, "generation": 0}

        def get_file(self, rpath, lpath):
            calls.append(rpath)
            shutil.copyfile(ckpt_path, lpath)

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: GenZeroFS())
    res = _load(str(ckpt_path), map_location="cpu")
    assert torch.all(res["weights"] == 3.0)
    assert len(calls) == 1


def test_load_remote_large_file_delegates_to_get_file_with_cache(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "large.ckpt"
    size = _big_checkpoint(ckpt_path)

    _use_tmp_cache_root(tmp_path, monkeypatch)
    get_file_calls = []
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size, calls=get_file_calls),
    )

    orig_load = torch.load
    load_kwargs = {}

    def spy_load(f, *args, **kwargs):
        load_kwargs.update(kwargs)
        return orig_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)

    # First call downloads via get_file into a unique staging file, then promotes it
    res = _load(str(ckpt_path), map_location="cpu")
    assert res["weights"].shape == (4096,)
    assert len(get_file_calls) == 1
    assert get_file_calls[0][0] == str(ckpt_path)
    assert f"checkpoint.ckpt.tmp.{os.getpid()}." in get_file_calls[0][1]
    if sys.platform != "win32":
        assert load_kwargs.get("mmap") is True
    else:
        assert "mmap" not in load_kwargs

    # Second call hits the cache and skips get_file entirely
    res2 = _load(str(ckpt_path), map_location="cpu")
    assert res2["weights"].shape == (4096,)
    assert len(get_file_calls) == 1


def test_load_remote_new_version_invalidates_same_size_cache(tmp_path, monkeypatch):
    """An overwritten checkpoint of identical size must not be served from the cache."""
    ckpt_path = tmp_path / "last.ckpt"
    size = _big_checkpoint(ckpt_path, fill=1.0)

    _use_tmp_cache_root(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size, version="epoch-1", calls=calls),
    )
    first = _load(str(ckpt_path), map_location="cpu")
    assert torch.all(first["weights"] == 1.0)
    assert len(calls) == 1

    # The trainer overwrites the same remote path; the size is unchanged but the content is not.
    new_size = _big_checkpoint(ckpt_path, fill=2.0)
    assert new_size == size
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, new_size, version="epoch-2", calls=calls),
    )

    second = _load(str(ckpt_path), map_location="cpu")
    assert torch.all(second["weights"] == 2.0), "stale cache served the previous version"
    assert len(calls) == 2

    # The superseded entry is reclaimed rather than left behind as a second full-size copy.
    prefix = _user_cache_prefix()
    entries = [d for d in os.listdir(tmp_path) if d.startswith(prefix) and not d.endswith(".lock")]
    assert len(entries) == 1


def test_load_remote_without_version_token_is_not_cached(tmp_path, monkeypatch):
    """Without a version token the cache cannot be invalidated safely, so stream instead."""
    ckpt_path = tmp_path / "unversioned.ckpt"
    size = _big_checkpoint(ckpt_path)

    _use_tmp_cache_root(tmp_path, monkeypatch)

    class NoVersionFS:
        def info(self, path):
            return {"size": size}

        def open(self, path, mode):
            return open(path, mode)

        def get_file(self, rpath, lpath):
            raise AssertionError("must not cache an object with no version token")

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: NoVersionFS())

    res = _load(str(ckpt_path), map_location="cpu")
    assert res["weights"].shape == (4096,)
    prefix = _user_cache_prefix()
    assert [d for d in os.listdir(tmp_path) if d.startswith(prefix)] == []


def test_load_legacy_non_zipfile_checkpoint(tmp_path, monkeypatch):
    """Checkpoints that cannot be memory-mapped must still load, locally and from the cache."""
    ckpt_path = tmp_path / "legacy.ckpt"
    torch.save({"weights": torch.tensor([7.0, 8.0])}, ckpt_path, _use_new_zipfile_serialization=False)

    # Local path
    loaded = _load(str(ckpt_path), map_location="cpu")
    torch.testing.assert_close(loaded["weights"], torch.tensor([7.0, 8.0]))

    # Remote path through the cache
    padded = tmp_path / "legacy_big.ckpt"
    torch.save(
        {"weights": torch.arange(4096, dtype=torch.float32)},
        padded,
        _use_new_zipfile_serialization=False,
    )
    size = os.path.getsize(padded)

    _use_tmp_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(padded, size),
    )
    res = _load(str(padded), map_location="cpu")
    assert res["weights"].shape == (4096,)


def test_load_remote_truncated_download_is_rejected(tmp_path, monkeypatch):
    """A short read must never be promoted into the cache."""
    ckpt_path = tmp_path / "truncated.ckpt"
    ckpt_path.write_bytes(b"x" * 2048)

    _use_tmp_cache_root(tmp_path, monkeypatch)

    class TruncatingFS:
        def info(self, path):
            return {"size": 4096, "etag": "v1"}

        def get_file(self, rpath, lpath):
            with open(lpath, "wb") as f:
                f.write(b"short")

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: TruncatingFS())

    with pytest.raises(OSError, match="Truncated download"):
        _load(str(ckpt_path), map_location="cpu")

    prefix = _user_cache_prefix()
    for d in os.listdir(tmp_path):
        if d.startswith(prefix) and not d.endswith(".lock"):
            assert not os.path.exists(os.path.join(tmp_path, d, "checkpoint.ckpt"))


def test_load_remote_cache_is_not_world_readable(tmp_path, monkeypatch):
    """A cached checkpoint bypasses storage ACLs, so it must stay inside one UID."""
    ckpt_path = tmp_path / "private.ckpt"
    size = _big_checkpoint(ckpt_path)

    _use_tmp_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size),
    )
    _load(str(ckpt_path), map_location="cpu")

    prefix = _user_cache_prefix()
    entries = [d for d in os.listdir(tmp_path) if d.startswith(prefix) and not d.endswith(".lock")]
    assert len(entries) == 1
    cache_dir = os.path.join(tmp_path, entries[0])
    dir_mode = os.stat(cache_dir).st_mode
    assert not dir_mode & 0o077, f"cache dir is group/world accessible: {oct(dir_mode & 0o777)}"
    file_mode = os.stat(os.path.join(cache_dir, "checkpoint.ckpt")).st_mode
    assert not file_mode & 0o077, f"cached checkpoint is group/world readable: {oct(file_mode & 0o777)}"


def test_load_remote_cleanup_on_exception(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "error.ckpt"
    ckpt_path.write_bytes(b"dummy")

    _use_tmp_cache_root(tmp_path, monkeypatch)

    class FailingFS:
        def info(self, path):
            return {"size": 4096, "etag": "v1"}

        def get_file(self, rpath, lpath):
            with open(lpath, "wb") as f:
                f.write(b"partial")
            raise RuntimeError("simulated download failure")

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: FailingFS())

    with pytest.raises(RuntimeError, match="simulated download failure"):
        _load(str(ckpt_path), map_location="cpu")

    prefix = _user_cache_prefix()
    for d in os.listdir(tmp_path):
        if d.startswith(prefix) and not d.endswith(".lock"):
            cache_dir = os.path.join(tmp_path, d)
            assert not os.path.exists(os.path.join(cache_dir, "checkpoint.ckpt"))
            assert not glob.glob(os.path.join(cache_dir, "checkpoint.ckpt.tmp.*"))


def test_load_remote_atomic_staging_recovery(tmp_path, monkeypatch):
    """An orphaned staging file from a killed process must be ignored, then reaped."""
    ckpt_path = tmp_path / "staged.ckpt"
    size = _big_checkpoint(ckpt_path)

    _use_tmp_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size),
    )

    path_digest = hashlib.sha256(str(ckpt_path).encode()).hexdigest()[:16]
    version_digest = hashlib.sha256(f"v1:{size}".encode()).hexdigest()[:16]
    cache_dir = tmp_path / f"{_user_cache_prefix()}{path_digest}_{version_digest}"
    # A killed process would have created this with 0o700; anything looser is treated as untrusted
    # and refused, so reproduce the real mode here.
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    orphan_tmp = cache_dir / "checkpoint.ckpt.tmp.999999.deadbeef"
    orphan_tmp.write_bytes(b"0" * 1024)

    res = _load(str(ckpt_path), map_location="cpu")
    assert res["weights"].shape == (4096,)
    assert (cache_dir / "checkpoint.ckpt").exists()
    assert os.path.getsize(cache_dir / "checkpoint.ckpt") == size
    # Staging files are checkpoint-sized, so leaving them behind would fill the cache root.
    assert not orphan_tmp.exists(), "the abandoned staging file was ignored but never reclaimed"


def test_load_remote_info_exception_fallback(tmp_path, monkeypatch):
    checkpoint = {"weights": torch.tensor([5.0])}
    ckpt_path = tmp_path / "fallback.ckpt"
    torch.save(checkpoint, ckpt_path)

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._is_local_file_protocol", lambda _: False)

    class ErrorFS:
        def info(self, path):
            raise FileNotFoundError("info not supported")

        def open(self, path, mode):
            return open(path, mode)

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: ErrorFS())
    loaded = _load(str(ckpt_path), map_location="cpu")
    torch.testing.assert_close(loaded["weights"], checkpoint["weights"])


@_requires_cache
def test_load_remote_shm_cache_and_repeat_hit_when_free_space_drops(tmp_path, monkeypatch):
    """Once cached in /dev/shm, subsequent loads must hit /dev/shm even if free space drops below 1.5x."""
    ckpt_path = tmp_path / "shm.ckpt"
    size = _big_checkpoint(ckpt_path)

    fake_shm = tmp_path / "fake_shm"
    fake_tmp = tmp_path / "fake_tmp"
    fake_shm.mkdir()
    fake_tmp.mkdir()

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._CACHE_MIN_SIZE_BYTES", 1024)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._is_local_file_protocol", lambda _: False)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._get_cache_roots", lambda: (str(fake_shm), str(fake_tmp)))

    calls = []
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size, calls=calls),
    )

    # Initially plenty of free space in fake_shm
    free_bytes = [size * 10]
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(total=size * 20, used=size, free=free_bytes[0]))

    res1 = _load(str(ckpt_path), map_location="cpu")
    assert res1["weights"].shape == (4096,)
    assert len(calls) == 1
    prefix = _user_cache_prefix()
    assert [d for d in os.listdir(fake_shm) if d.startswith(prefix) and not d.endswith(".lock")]
    assert [d for d in os.listdir(fake_tmp) if d.startswith(prefix)] == []

    # Simulate free space dropping below 1.5 * size after the first download
    free_bytes[0] = int(size * 0.5)
    res2 = _load(str(ckpt_path), map_location="cpu")
    assert res2["weights"].shape == (4096,)
    # Must hit existing fake_shm cache and NOT download a duplicate into fake_tmp
    assert len(calls) == 1
    assert [d for d in os.listdir(fake_tmp) if d.startswith(prefix)] == []


def test_clear_cache_removes_entries(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "purge.ckpt"
    size = _big_checkpoint(ckpt_path)

    _use_tmp_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size),
    )
    _load(str(ckpt_path), map_location="cpu")
    prefix = _user_cache_prefix()
    assert [d for d in os.listdir(tmp_path) if d.startswith(prefix)]

    clear_cache()
    remaining = [d for d in os.listdir(tmp_path) if d.startswith(prefix)]
    # The payload directories are gone. Only the empty lock markers survive: unlinking one is unsafe
    # while a peer may be blocked on it, and they cost an inode each.
    assert all(d.endswith(".lock") for d in remaining), remaining
    assert all(os.path.getsize(os.path.join(tmp_path, d)) == 0 for d in remaining)


def _mp_worker(ckpt_path_str, cache_root_str, counter_dir_str, size, barrier):
    from lightning.fabric.utilities import cloud_io

    class MPStubFS:
        def info(self, path):
            return {"size": size, "etag": "mp-v1"}

        def get_file(self, rpath, lpath):
            (Path(counter_dir_str) / f"call.{os.getpid()}").write_text(lpath)
            shutil.copyfile(rpath, lpath)

    cloud_io._CACHE_MIN_SIZE_BYTES = 1024
    cloud_io._is_local_file_protocol = lambda _: False
    cloud_io._get_cache_roots = lambda: (cache_root_str,)
    cloud_io.get_filesystem = lambda _: MPStubFS()

    barrier.wait()
    res = cloud_io._load(ckpt_path_str, map_location="cpu")
    assert res["weights"].shape == (4096,)


@pytest.mark.skipif(sys.platform != "linux", reason="fork-based barrier test requires Linux")
def test_load_remote_multiprocess_singleflight(tmp_path):
    """Multiple processes loading the same remote checkpoint concurrently must download it exactly once."""
    ckpt_path = tmp_path / "mp.ckpt"
    size = _big_checkpoint(ckpt_path, fill=4.0)
    cache_root = tmp_path / "mp_cache"
    counter_dir = tmp_path / "mp_calls"
    cache_root.mkdir()
    counter_dir.mkdir()

    ctx = mp.get_context("fork")
    nprocs = 4
    barrier = ctx.Barrier(nprocs)
    procs = [
        ctx.Process(target=_mp_worker, args=(str(ckpt_path), str(cache_root), str(counter_dir), size, barrier))
        for _ in range(nprocs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    assert [p.exitcode for p in procs] == [0] * nprocs
    assert len(list(counter_dir.glob("call.*"))) == 1


def test_remote_version_requires_a_strong_validator():
    """A modification time is too coarse to invalidate a cache on, so it must not be accepted."""
    assert _remote_version({"etag": "abc"}) == "abc"
    assert _remote_version({"generation": 0}) == "0"
    assert _remote_version({"version_id": "v7"}) == "v7"
    assert _remote_version({"mtime": 1700000000, "LastModified": "now", "size": 1}) == ""


@_requires_cache
def test_cache_can_be_disabled_by_env(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "opt_out.ckpt"
    size = _big_checkpoint(ckpt_path)

    _use_tmp_cache_root(tmp_path, monkeypatch)
    monkeypatch.setenv("LIGHTNING_CHECKPOINT_CACHE", "0")
    calls = []
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size, calls=calls),
    )

    res = _load(str(ckpt_path), map_location="cpu")
    assert res["weights"].shape == (4096,)
    assert calls == [], "the checkpoint was cached even though the cache is disabled"
    assert [d for d in os.listdir(tmp_path) if d.startswith(_user_cache_prefix())] == []


@_requires_cache
def test_cache_root_can_be_overridden_by_env(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "override.ckpt"
    size = _big_checkpoint(ckpt_path)
    cache_root = tmp_path / "custom_cache" / "nested"

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._CACHE_MIN_SIZE_BYTES", 1024)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._is_local_file_protocol", lambda _: False)
    monkeypatch.setenv("LIGHTNING_CHECKPOINT_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size),
    )

    res = _load(str(ckpt_path), map_location="cpu")
    assert res["weights"].shape == (4096,)
    assert [d for d in os.listdir(cache_root) if d.startswith(_user_cache_prefix())]


@_requires_cache
def test_cache_keeps_existing_entries_and_streams_when_root_is_full(tmp_path, monkeypatch):
    """Nothing is evicted to make room: a full root stops accepting entries and streams instead."""
    first = tmp_path / "a.ckpt"
    second = tmp_path / "b.ckpt"
    size = _big_checkpoint(first, fill=1.0)
    _big_checkpoint(second, fill=2.0)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    _use_tmp_cache_root(tmp_path, monkeypatch)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._get_cache_roots", lambda: (str(cache_root),))

    free = [size * 10]
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(total=size * 20, used=0, free=free[0]))

    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(first, size),
    )
    assert torch.all(_load(str(first), map_location="cpu")["weights"] == 1.0)

    # The root fills up, so the second checkpoint must stream rather than evict the first.
    free[0] = 0
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(second, size),
    )
    assert torch.all(_load(str(second), map_location="cpu")["weights"] == 2.0)

    prefix = _user_cache_prefix()
    entries = [d for d in os.listdir(cache_root) if d.startswith(prefix) and not d.endswith(".lock")]
    assert len(entries) == 1, f"an existing entry was evicted to make room: {entries}"


@_requires_cache
def test_reclamation_skips_an_entry_another_process_is_using(tmp_path, monkeypatch):
    """Reclaiming an entry mid-download would delete a peer's staging file out from under it."""
    import fcntl

    ckpt_path = tmp_path / "busy.ckpt"
    size = _big_checkpoint(ckpt_path)
    _use_tmp_cache_root(tmp_path, monkeypatch)

    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size, version="v1"),
    )
    _load(str(ckpt_path), map_location="cpu")

    prefix = _user_cache_prefix()
    entry = next(d for d in os.listdir(tmp_path) if d.startswith(prefix) and not d.endswith(".lock"))
    old_dir = tmp_path / entry

    # Take the lock the way a concurrent downloader on this node would.
    fd = os.open(f"{old_dir}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        monkeypatch.setattr(
            "lightning.fabric.utilities.cloud_io.get_filesystem",
            lambda _: _versioned_fs(ckpt_path, size, version="v2"),
        )
        _load(str(ckpt_path), map_location="cpu")
        assert old_dir.exists(), "reclamation deleted an entry another process was using"

        clear_cache()
        assert old_dir.exists(), "clear_cache deleted an entry another process was using"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    clear_cache()
    assert not old_dir.exists()


@_requires_cache
def test_failed_download_keeps_the_lock_file(tmp_path, monkeypatch):
    """Unlinking a lock we hold orphans the inode and lets two ranks download at once."""
    ckpt_path = tmp_path / "boom.ckpt"
    _use_tmp_cache_root(tmp_path, monkeypatch)

    class FailingFS:
        def info(self, path):
            return {"size": 4096, "etag": "v1"}

        def get_file(self, rpath, lpath):
            raise RuntimeError("simulated download failure")

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: FailingFS())

    with pytest.raises(RuntimeError, match="simulated download failure"):
        _load(str(ckpt_path), map_location="cpu")

    prefix = _user_cache_prefix()
    locks = [d for d in os.listdir(tmp_path) if d.startswith(prefix) and d.endswith(".lock")]
    assert len(locks) == 1


@_requires_cache
def test_cache_dir_we_do_not_own_is_not_used(tmp_path, monkeypatch):
    """/dev/shm is world-writable, so a pre-created entry could feed a swapped checkpoint to pickle."""
    ckpt_path = tmp_path / "hijack.ckpt"
    size = _big_checkpoint(ckpt_path)
    _use_tmp_cache_root(tmp_path, monkeypatch)

    path_digest = hashlib.sha256(str(ckpt_path).encode()).hexdigest()[:16]
    version_digest = hashlib.sha256(f"v1:{size}".encode()).hexdigest()[:16]
    cache_dir = tmp_path / f"{_user_cache_prefix()}{path_digest}_{version_digest}"

    # Stand in for an attacker-planted symlink: `os.makedirs(exist_ok=True)` accepts it and the
    # follow-up `chmod` resolves it, so only an `lstat` check catches it.
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()
    cache_dir.symlink_to(attacker_dir, target_is_directory=True)

    calls = []
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size, calls=calls),
    )

    res = _load(str(ckpt_path), map_location="cpu")
    assert res["weights"].shape == (4096,)
    assert calls == [], "the checkpoint was written into a directory we do not exclusively own"
    assert not (attacker_dir / "checkpoint.ckpt").exists()


@_requires_cache
def test_load_remote_streams_when_no_root_has_room(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "toobig.ckpt"
    size = _big_checkpoint(ckpt_path)
    _use_tmp_cache_root(tmp_path, monkeypatch)

    calls = []
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size, calls=calls),
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(total=size * 20, used=size * 20, free=0))

    res = _load(str(ckpt_path), map_location="cpu")
    assert res["weights"].shape == (4096,)
    assert calls == []
    assert [d for d in os.listdir(tmp_path) if d.startswith(_user_cache_prefix())] == []


@_requires_cache
def test_decoded_file_larger_than_reported_size_is_a_cache_hit(tmp_path, monkeypatch):
    """A gzip-transcoded object decodes to more bytes than `fs.info` reports."""
    ckpt_path = tmp_path / "transcoded.ckpt"
    size = _big_checkpoint(ckpt_path)
    _use_tmp_cache_root(tmp_path, monkeypatch)

    calls = []

    class TranscodingFS:
        def info(self, path):
            # The object is stored compressed, so the reported size is smaller than what we get.
            return {"size": size // 2, "etag": "v1"}

        def get_file(self, rpath, lpath):
            calls.append(rpath)
            shutil.copyfile(ckpt_path, lpath)

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: TranscodingFS())

    assert _load(str(ckpt_path), map_location="cpu")["weights"].shape == (4096,)
    assert len(calls) == 1
    assert _load(str(ckpt_path), map_location="cpu")["weights"].shape == (4096,)
    assert len(calls) == 1, "the decoded checkpoint was re-downloaded instead of being reused"


@_requires_cache
def test_cached_checkpoint_survives_a_concurrent_reclamation(tmp_path, monkeypatch):
    """A peer reclaiming the entry while we load it must not delete the file out from under us."""
    ckpt_path = tmp_path / "locked_load.ckpt"
    size = _big_checkpoint(ckpt_path, fill=4.0)
    _use_tmp_cache_root(tmp_path, monkeypatch)

    orig_torch_load = cloud_io._torch_load

    def load_racing_a_peer(path, map_location, weights_only):
        # Stand in for a peer process that just cached a newer version of the same path and is
        # reclaiming the superseded entries. It must skip ours because we are still inside it.
        cloud_io._remove_cache_entry(os.path.dirname(path))
        return orig_torch_load(path, map_location, weights_only)

    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._torch_load", load_racing_a_peer)
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size),
    )

    # Without the load happening under the entry lock this raises FileNotFoundError.
    assert torch.all(_load(str(ckpt_path), map_location="cpu")["weights"] == 4.0)


@_requires_cache
def test_get_cache_roots_does_not_create_directory(tmp_path, monkeypatch):
    """_get_cache_roots() must not create directories; _load creates them on demand."""
    custom_root = tmp_path / "deferred" / "cache_dir"
    monkeypatch.setenv("LIGHTNING_CHECKPOINT_CACHE_DIR", str(custom_root))
    assert not custom_root.exists()

    roots = cloud_io._get_cache_roots()
    assert roots == (str(custom_root),)
    assert not custom_root.exists(), "_get_cache_roots mutated the filesystem"

    # clear_cache must not create custom_root either
    cloud_io.clear_cache()
    assert not custom_root.exists(), "clear_cache mutated the filesystem"

    # _load creates custom_root when writing
    ckpt_path = tmp_path / "test.ckpt"
    size = _big_checkpoint(ckpt_path)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._CACHE_MIN_SIZE_BYTES", 1024)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._is_local_file_protocol", lambda _: False)
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.get_filesystem",
        lambda _: _versioned_fs(ckpt_path, size),
    )
    _load(str(ckpt_path), map_location="cpu")
    assert custom_root.exists()


@_requires_cache
def test_stale_unlocked_lock_file_does_not_pin_full_root(tmp_path, monkeypatch):
    """An unlocked .lock left behind by a failed download must not pin a full primary root."""
    ckpt_path = tmp_path / "retry_after_failure.ckpt"
    size = _big_checkpoint(ckpt_path)

    shm_root = tmp_path / "shm"
    disk_root = tmp_path / "disk"
    shm_root.mkdir()
    disk_root.mkdir()

    monkeypatch.delenv("LIGHTNING_CHECKPOINT_CACHE", raising=False)
    monkeypatch.delenv("LIGHTNING_CHECKPOINT_CACHE_DIR", raising=False)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._CACHE_MIN_SIZE_BYTES", 1024)
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io._is_local_file_protocol", lambda _: False)
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io._get_cache_roots",
        lambda: (str(shm_root), str(disk_root)),
    )

    # First attempt: shm_root has plenty of space, but the download fails mid-flight, leaving an
    # unlocked `<entry>.lock` file in shm_root.
    class FailOnceFS:
        def __init__(self):
            self.attempts = 0
            self._inner = _versioned_fs(ckpt_path, size)

        def info(self, path):
            return self._inner.info(path)

        def get_file(self, rpath, lpath):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("transient network error")
            return self._inner.get_file(rpath, lpath)

        def open(self, path, mode="rb"):
            return self._inner.open(path, mode)

    fs = FailOnceFS()
    monkeypatch.setattr("lightning.fabric.utilities.cloud_io.get_filesystem", lambda _: fs)

    with pytest.raises(OSError, match="transient network error"):
        _load(str(ckpt_path), map_location="cpu")

    shm_locks = [f for f in os.listdir(shm_root) if f.endswith(".lock")]
    assert len(shm_locks) == 1, "Failed download should have created the lock file in shm_root"

    # Now shm_root fills up (free = 0). The leftover unlocked .lock in shm_root must NOT trick
    # root selection into picking shm_root and bypassing the free-space check; the retry must fall
    # through to disk_root.
    from collections import namedtuple

    Usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        "lightning.fabric.utilities.cloud_io.shutil.disk_usage",
        lambda path: (
            Usage(size * 10, size * 10, 0) if str(path).startswith(str(shm_root)) else Usage(size * 10, 0, size * 10)
        ),
    )

    loaded = _load(str(ckpt_path), map_location="cpu")
    assert loaded["weights"].shape == (4096,)

    prefix = _user_cache_prefix()
    disk_entries = [d for d in os.listdir(disk_root) if d.startswith(prefix) and not d.endswith(".lock")]
    assert len(disk_entries) == 1
    assert os.path.exists(os.path.join(disk_root, disk_entries[0], "checkpoint.ckpt"))
