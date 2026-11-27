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
"""Utilities related to data saving/loading."""

import contextlib
import errno
import getpass
import glob
import hashlib
import importlib
import io
import logging
import os
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Optional, Union

import fsspec
import fsspec.utils
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from lightning_utilities.core.imports import module_available

from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH

log = logging.getLogger(__name__)

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False

# Streaming a small checkpoint straight from the object store beats paying for a local copy.
_CACHE_MIN_SIZE_BYTES = 128 * 1024 * 1024
_CACHE_DIR_PREFIX = "lightning_cache_"
_CACHE_FILE_NAME = "checkpoint.ckpt"
_CACHE_ENABLED_ENV = "LIGHTNING_CHECKPOINT_CACHE"
_CACHE_DIR_ENV = "LIGHTNING_CHECKPOINT_CACHE_DIR"
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _cache_enabled() -> bool:
    """Return whether remote checkpoints may be cached on the node."""
    if not _HAS_FCNTL:  # pragma: no cover
        # Without an advisory lock every rank downloads the same object and they race on
        # `os.replace`, which fails on Windows while a peer still has the target open.
        return False
    return os.environ.get(_CACHE_ENABLED_ENV, "1") != "0"


@contextlib.contextmanager
def _entry_lock(lock_path: str, blocking: bool = True) -> Iterator[tuple[Optional[int], bool]]:
    """Hold the node-local advisory lock guarding one cache entry, using stdlib ``fcntl.flock``.

    Yields ``(fd, exclusive)``. The first caller into the entry takes it exclusively (``LOCK_EX``)
    and gets ``exclusive=True``; blocking callers that find it taken instead wait on ``LOCK_SH`` and
    resume together with ``exclusive=False`` once the downloader downgrades before loading. A
    non-blocking attempt yields ``(None, False)`` when another process is already inside the entry.

    """
    if not _HAS_FCNTL:  # pragma: no cover
        yield None, True
        return
    fd: Optional[int] = None
    exclusive = False
    try:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | _O_NOFOLLOW, 0o600)
        except OSError as e:
            log.debug(f"Cannot open the checkpoint cache lock {lock_path} ({e}).")
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                exclusive = True
            except OSError:
                if not blocking:
                    os.close(fd)
                    fd = None
                else:
                    # Wait for the downloader to finish and downgrade, then load alongside it.
                    fcntl.flock(fd, fcntl.LOCK_SH)
        yield fd, exclusive
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _is_entry_active(lock_path: str) -> bool:
    """Return whether another process currently holds ``lock_path``."""
    if not os.path.exists(lock_path):
        return False
    with _entry_lock(lock_path, blocking=False) as (fd, _):
        return fd is None


def _remove_cache_entry(cache_dir: str) -> None:
    """Delete a cache entry's payload, skipping it while another process is using it.

    The empty lock file is deliberately left behind. Unlinking it would let a peer that is blocked
    in ``flock`` wake up holding an inode with no name, while a newcomer creates and locks the
    replacement file, so both would believe they own the entry and one could delete the directory
    out from under the other.

    """
    with _entry_lock(f"{cache_dir}.lock", blocking=False) as (fd, _):
        if _HAS_FCNTL and fd is None:
            log.debug(f"Not reclaiming {cache_dir}: another process is still using it.")
            return
        shutil.rmtree(cache_dir, ignore_errors=True)


def _get_cache_roots() -> tuple[str, ...]:
    """Return candidate cache root directories in order of preference."""
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return (override,)
    return ("/dev/shm", tempfile.gettempdir())


def _user_cache_prefix() -> str:
    """Namespace cache directories by UID so multi-user nodes never collide in /dev/shm or /tmp."""
    if hasattr(os, "getuid"):
        return f"{_CACHE_DIR_PREFIX}{os.getuid()}_"
    # `getpass.getuser()` can return "DOMAIN\\user", which is not a usable path component.
    return f"{_CACHE_DIR_PREFIX}{hashlib.sha256(getpass.getuser().encode()).hexdigest()[:16]}_"


def _is_private_dir(path: str) -> bool:
    """Return whether ``path`` is a real directory owned by us that nobody else can write to.

    ``/dev/shm`` and ``/tmp`` are world-writable, so another local user can pre-create the entry (or
    a symlink standing in for it) and swap the checkpoint before we unpickle it.

    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o022:
        return False
    return not hasattr(os, "getuid") or info.st_uid == os.getuid()


def _cache_entries(root: str) -> Iterator[str]:
    """Yield this user's cache entry directories in ``root``, with any ``.lock`` suffix stripped."""
    seen = set()
    for match in glob.glob(os.path.join(root, f"{_user_cache_prefix()}*")):
        entry = match[: -len(".lock")] if match.endswith(".lock") else match
        if entry not in seen:
            seen.add(entry)
            yield entry


def _torch_load(
    path: str,
    map_location: _MAP_LOCATION_TYPE,
    weights_only: Optional[bool],
) -> Any:
    """Load a local checkpoint, memory-mapping it when the file format allows."""
    if sys.platform != "win32":
        try:
            return torch.load(
                path,
                map_location=map_location,  # type: ignore[arg-type]
                weights_only=weights_only,
                mmap=True,
            )
        except (RuntimeError, ValueError) as e:
            if "mmap" not in str(e):
                raise
            log.debug(f"Checkpoint {path} cannot be memory-mapped ({e}); loading it normally.")
    return torch.load(
        path,
        map_location=map_location,  # type: ignore[arg-type]
        weights_only=weights_only,
    )


def _remote_version(file_info: dict[str, Any]) -> str:
    """Return a token that changes whenever the remote object's content changes.

    Only strong validators are accepted. A modification time is too coarse to key a cache on: an overwrite within the
    same second that keeps the size is indistinguishable from the original, so the stale weights would be served.
    Objects without one are streamed instead of cached.

    """
    for key in ("etag", "ETag", "generation", "version_id", "VersionId"):
        value = file_info.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def _cached_file_is_complete(path: str, remote_size: int) -> bool:
    """Return whether ``path`` holds a fully downloaded copy of an object of ``remote_size`` bytes.

    Only a short file indicates a partial download. Objects served with ``Content-Encoding: gzip``
    decode to more bytes than ``fs.info`` reports, and they must not be re-downloaded on every load.

    """
    try:
        return os.path.getsize(path) >= remote_size
    except OSError:
        return False


def _reclaim_superseded_entries(path_digest: str, keep: str) -> None:
    """Delete older cache entries for the same remote path across candidate roots."""
    keep_abs = os.path.abspath(keep)
    prefix = f"{_user_cache_prefix()}{path_digest}_"
    for root in _get_cache_roots():
        for stale in _cache_entries(root):
            if os.path.basename(stale).startswith(prefix) and os.path.abspath(stale) != keep_abs:
                _remove_cache_entry(stale)


def clear_cache() -> None:
    """Remove every local checkpoint cache entry written by :func:`_load` for the current user.

    Entries that another process is currently downloading into or loading from are left alone. The
    empty ``.lock`` marker files are kept, since removing one is unsafe while a peer may be waiting
    on it; they cost an inode each and are cleared when the root is (``/dev/shm`` on reboot).

    """
    for root in _get_cache_roots():
        for entry in _cache_entries(root):
            _remove_cache_entry(entry)


def _try_cached_load(
    fs: AbstractFileSystem,
    path_str: str,
    map_location: _MAP_LOCATION_TYPE,
    weights_only: Optional[bool],
) -> tuple[bool, Any]:
    """Attempt to stage ``path_str`` in the node-local checkpoint cache and load it from disk.

    Returns ``(True, checkpoint)`` when the checkpoint was loaded from the local cache, or
    ``(False, None)`` when the caller should fall back to streaming from ``fs``.

    """
    if not _cache_enabled():
        return False, None

    try:
        file_info = fs.info(path_str)
        raw_size = file_info.get("size")
        file_size = int(raw_size) if raw_size is not None else 0
    except Exception as e:
        log.debug(f"Cannot stat {path_str} ({e}); streaming it rather than caching it locally.")
        return False, None

    # Stream small files: a local copy costs more than it saves.
    if file_size < _CACHE_MIN_SIZE_BYTES:
        return False, None

    # Without a version token the cache cannot be invalidated, and caching on size alone would
    # serve stale bytes for an overwritten checkpoint.
    version = _remote_version(file_info)
    if not version:
        log.debug(
            f"{path_str} exposes no version token (etag/generation/version_id), so it is streamed"
            f" rather than cached locally."
        )
        return False, None

    path_digest = hashlib.sha256(path_str.encode()).hexdigest()[:16]
    version_digest = hashlib.sha256(f"{version}:{file_size}".encode()).hexdigest()[:16]
    cache_subdir = f"{_user_cache_prefix()}{path_digest}_{version_digest}"
    roots = _get_cache_roots()

    # Prefer a root that already holds the entry, or that a peer rank is actively downloading/loading
    # in, so that everyone converges on a single copy instead of racing to fill two roots.
    selected_root: Optional[str] = None
    for root in roots:
        candidate_dir = os.path.join(root, cache_subdir)
        if _cached_file_is_complete(os.path.join(candidate_dir, _CACHE_FILE_NAME), file_size) or _is_entry_active(
            f"{candidate_dir}.lock"
        ):
            selected_root = root
            break

    if selected_root is None:
        for root in roots:
            if not os.path.isdir(root):
                with contextlib.suppress(OSError):
                    os.makedirs(root, mode=0o700, exist_ok=True)
            if not (os.path.isdir(root) and os.access(root, os.W_OK)):
                continue
            try:
                # Free space is the only admission check: nothing is ever evicted to make room, so
                # a filling root simply stops accepting new entries and later loads stream instead.
                # Headroom on top of the payload itself: peer ranks on the node may be staging
                # their own checkpoints into the same root at the same time.
                if shutil.disk_usage(root).free < file_size * 1.5:
                    continue
            except OSError:
                continue
            selected_root = root
            break

    if selected_root is None:
        log.warning(f"No cache root has room for {path_str} ({file_size} bytes); streaming it instead.")
        return False, None

    cache_dir = os.path.join(selected_root, cache_subdir)
    try:
        # Deliberately no `os.chmod` here. The roots are world-writable, so another local user can
        # plant a symlink at `cache_dir`; `os.chmod` follows symlinks and would apply the mode to
        # whatever the link points at. `mkdir` already caps the new directory at 0o700 (a umask can
        # only clear bits, never add them), and `_is_private_dir` below rejects anything we did not
        # create ourselves.
        os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    except OSError as e:
        log.warning(f"Cannot prepare the checkpoint cache directory {cache_dir} ({e}); streaming instead.")
        return False, None

    if not _is_private_dir(cache_dir):
        log.warning(
            f"Refusing to cache {path_str} in {cache_dir}: it is not a directory owned exclusively by this"
            f" user, so its contents cannot be trusted. Streaming the checkpoint instead."
        )
        return False, None

    local_path = os.path.join(cache_dir, _CACHE_FILE_NAME)
    lock_path = f"{cache_dir}.lock"
    staging_path = os.path.join(cache_dir, f"{_CACHE_FILE_NAME}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")

    downloaded = False
    with _entry_lock(lock_path) as (fd, exclusive):
        if _HAS_FCNTL and fd is None:  # pragma: no cover
            log.warning(f"Cannot lock the checkpoint cache entry {cache_dir}; streaming instead.")
            return False, None

        # A loop, not a single check: if the rank holding LOCK_EX dies mid-download, every waiter
        # wakes with the file still missing and one of them has to take over.
        while not _cached_file_is_complete(local_path, file_size):
            if _HAS_FCNTL and fd is not None and not exclusive:
                # The downloader died before promoting the file, so try to take over. The request
                # must be non-blocking: a blocking LOCK_EX is not woken by the next downloader's
                # downgrade to LOCK_SH, so every waiter would end up unpickling in series rather
                # than together, which is the whole point of the downgrade.
                fcntl.flock(fd, fcntl.LOCK_UN)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    exclusive = True
                except OSError:
                    # Another waiter got there first; park until it downgrades, then re-check.
                    fcntl.flock(fd, fcntl.LOCK_SH)
                    continue

            size_gb = file_size / (1024**3)
            log.info(f"Fetching {path_str} ({size_gb:.2f} GB) to {staging_path}...")

            # A rank killed mid-download leaves its staging file behind, and nothing else ever
            # reaps them even though each one is the size of a whole checkpoint. We hold the entry
            # exclusively here, so anything still staged is abandoned.
            for abandoned in glob.glob(os.path.join(cache_dir, f"{_CACHE_FILE_NAME}.tmp.*")):
                with contextlib.suppress(OSError):
                    os.remove(abandoned)

            try:
                fs.get_file(path_str, staging_path)
                staged_size = os.path.getsize(staging_path)
                if staged_size < file_size:
                    raise OSError(f"Truncated download of {path_str}: expected {file_size} bytes, got {staged_size}")
                os.chmod(staging_path, 0o600)
                os.replace(staging_path, local_path)
                downloaded = True
            finally:
                # Only ever remove our own staging file, never the promoted checkpoint.
                with contextlib.suppress(OSError):
                    os.remove(staging_path)

        # Downgrade LOCK_EX -> LOCK_SH so all waiting peer ranks can unpickle concurrently while
        # still blocking concurrent _remove_cache_entry (which requires LOCK_EX | LOCK_NB).
        if _HAS_FCNTL and fd is not None and exclusive:
            fcntl.flock(fd, fcntl.LOCK_SH)

        checkpoint = _torch_load(local_path, map_location, weights_only)

    if downloaded:
        _reclaim_superseded_entries(path_digest, cache_dir)

    return True, checkpoint


def _load(
    path_or_url: Union[IO, _PATH],
    map_location: _MAP_LOCATION_TYPE = None,
    weights_only: Optional[bool] = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
        weights_only: If ``True``, restricts loading to ``state_dicts`` of plain ``torch.Tensor`` and other primitive
            types. If loading a checkpoint from a trusted source that contains an ``nn.Module``, use
            ``weights_only=False``. If loading checkpoint from an untrusted source, we recommend using
            ``weights_only=True``. For more information, please refer to the
            `PyTorch Developer Notes on Serialization Semantics <https://docs.pytorch.org/docs/main/notes/serialization.html#id3>`_.

    """
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(
            path_or_url,
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
            weights_only=weights_only,
        )

    path_str = str(path_or_url)
    if path_str.startswith("http"):
        if weights_only is None:
            weights_only = False
            log.debug(
                f"Defaulting to `weights_only=False` for remote checkpoint: {path_or_url}."
                f" If loading a checkpoint from an untrusted source, we recommend using `weights_only=True`."
            )

        return torch.hub.load_state_dict_from_url(
            path_str,
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
        )

    fs = get_filesystem(path_or_url)

    # 1. Local path optimization: no copy is needed, map the file directly.
    if _is_local_file_protocol(path_str):
        _, local_file = url_to_fs(path_str)
        return _torch_load(local_file, map_location, weights_only)

    # 2. Remote checkpoint fetching via stdlib fcntl.flock + fs.get_file
    hit, checkpoint = _try_cached_load(fs, path_str, map_location, weights_only)
    if hit:
        return checkpoint

    with fs.open(path_str, "rb") as f:
        return torch.load(
            f,
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
        )


def get_filesystem(path: _PATH, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


def _atomic_save(checkpoint: dict[str, Any], filepath: _PATH) -> None:
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.

    """
    log.debug(f"Saving checkpoint: {filepath}")

    try:
        # We use a transaction here to avoid file corruption if the save gets interrupted
        fs, urlpath = fsspec.core.url_to_fs(str(filepath))
        with fs.transaction:
            if _is_object_storage(fs):
                is_azure = False
                if module_available("adlfs"):
                    from adlfs import AzureBlobFileSystem

                    is_azure = isinstance(fs, AzureBlobFileSystem)

                # Object storage cannot stream `torch.save`, so build the payload in memory first.
                bytesbuffer = io.BytesIO()
                torch.save(checkpoint, bytesbuffer)
                if is_azure:
                    # Azure uses a plain write because adlfs stages blocks sequentially, making
                    # pipe() slower.
                    with fs.open(urlpath, "wb") as f:
                        f.write(bytesbuffer.getvalue())
                else:
                    # Use fs.pipe() for S3/GCS where it triggers parallel multipart uploads,
                    # giving 4-5x throughput improvement for checkpoints >= 500 MB.
                    fs.pipe(urlpath, bytesbuffer.getvalue())
            else:
                # Stream directly to the file so we never hold a second full copy of the checkpoint
                # in memory. This matters for large FSDP/ModelParallel full state dicts on local disk.
                with fs.open(urlpath, "wb") as f:
                    torch.save(checkpoint, f)
    except PermissionError as e:
        if isinstance(e.__context__, OSError) and getattr(e.__context__, "errno", None) == errno.EXDEV:
            raise RuntimeError(
                'Upgrade fsspec to enable cross-device local checkpoints: pip install "fsspec[http]>=2025.5.0"',
            ) from e
        raise


def _is_object_storage(fs: AbstractFileSystem) -> bool:
    if module_available("adlfs"):
        from adlfs import AzureBlobFileSystem

        if isinstance(fs, AzureBlobFileSystem):
            return True

    if module_available("gcsfs"):
        from gcsfs import GCSFileSystem

        if isinstance(fs, GCSFileSystem):
            return True

    if module_available("s3fs"):
        from s3fs import S3FileSystem

        if isinstance(fs, S3FileSystem):
            return True

    return False


def _is_dir(fs: AbstractFileSystem, path: Union[str, Path], strict: bool = False) -> bool:
    """Check if a path is directory-like.

    This function determines if a given path is considered directory-like, taking into account the behavior
    specific to object storage platforms. For other filesystems, it behaves similarly to the standard `fs.isdir`
    method.

    Args:
        fs: The filesystem to check the path against.
        path: The path or URL to be checked.
        strict: A flag specific to Object Storage platforms. If set to ``False``, any non-existing path is considered
            as a valid directory-like path. In such cases, the directory (and any non-existing parent directories)
            will be created on the fly. Defaults to False.

    """
    # Object storage fsspec's are inconsistent with other file systems because they do not have real directories,
    # see for instance https://gcsfs.readthedocs.io/en/latest/api.html?highlight=makedirs#gcsfs.core.GCSFileSystem.mkdir
    # In particular, `fs.makedirs` is a no-op so we use `strict=False` to consider any path as valid, except if the
    # path already exists but is a file
    if _is_object_storage(fs):
        if strict:
            return fs.isdir(path)

        # Check if the path is not already taken by a file. If not, it is considered a valid directory-like path
        # because the directory (and all non-existing parent directories) will be created on the fly.
        return not fs.isfile(path)

    return fs.isdir(path)


def _is_local_file_protocol(path: _PATH) -> bool:
    return fsspec.utils.get_protocol(str(path)) == "file"


def _resolve_path(path: _PATH) -> Union[str, Path]:
    """Return a ``Path`` for local file paths and a plain ``str`` for remote fsspec URLs.

    ``Path()`` collapses the double slash in a URL (e.g. ``gs://bucket`` -> ``gs:/bucket``),
    corrupting it, so remote URLs must be kept as strings.

    """
    if _is_local_file_protocol(str(path)):
        _, urlpath = url_to_fs(str(path))
        return Path(urlpath).expanduser().resolve()
    return str(path)


def _checkpoint_join(path: Union[str, Path], name: str) -> Union[str, Path]:
    """Join ``name`` onto a checkpoint ``path`` without corrupting remote URLs."""
    if isinstance(path, Path):
        return path / name

    # Remote URLs stay as strings because `Path`/`os.path.join` use local
    # filesystem semantics (e.g. '\' on Windows), which can corrupt URLs.
    return str(path).rstrip("/") + "/" + name


def _is_checkpoint_dir(path: Union[str, Path]) -> bool:
    """Return whether ``path`` points to an existing directory, supporting fsspec paths."""
    if isinstance(path, Path):
        return path.is_dir()
    return get_filesystem(path).isdir(str(path))


def _prepare_directory_checkpoint(path: Union[str, Path]) -> None:
    """Ensure ``path`` is a directory for a sharded checkpoint.

    Removes a conflicting file sitting at ``path`` and creates the directory. Creating a
    directory is a no-op on object storage, which has no real directories.

    """
    if isinstance(path, Path):
        if path.is_file():
            path.unlink()
        path.mkdir(parents=True, exist_ok=True)
        return
    fs = get_filesystem(path)
    if fs.isfile(str(path)):
        fs.rm(str(path))
    if not _is_object_storage(fs):
        fs.makedirs(str(path), exist_ok=True)


def _remove_checkpoint(path: Union[str, Path]) -> None:
    """Remove a checkpoint file or directory (recursively), supporting fsspec paths."""
    if isinstance(path, Path):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        return
    fs = get_filesystem(path)
    if fs.exists(str(path)):
        fs.rm(str(path), recursive=True)


def _get_distributed_checkpoint_writer(path: _PATH) -> Any:
    if _is_local_file_protocol(str(path)):
        from torch.distributed.checkpoint import FileSystemWriter

        # FSDP's FileSystemWriter streams the tensors to disk to minimize memory peaks
        return FileSystemWriter(path=path, single_file_per_rank=True)
    FsspecWriter = _import_fsspec_dcp_filesystem("FsspecWriter")
    return FsspecWriter(path=str(path), single_file_per_rank=True)


def _get_distributed_checkpoint_reader(path: _PATH) -> Any:
    if _is_local_file_protocol(str(path)):
        from torch.distributed.checkpoint import FileSystemReader

        return FileSystemReader(path=path)
    FsspecReader = _import_fsspec_dcp_filesystem("FsspecReader")
    return FsspecReader(path=str(path))


def _import_fsspec_dcp_filesystem(name: str) -> Any:
    """Import ``FsspecReader``/``FsspecWriter`` from torch's private DCP fsspec module.

    These live in a private module that not every PyTorch build ships, so raise an actionable error
    instead of letting a bare ``ImportError`` surface from deep in the call stack.

    """
    try:
        module = importlib.import_module("torch.distributed.checkpoint._fsspec_filesystem")
    except ImportError as e:
        raise ImportError(
            "Remote (fsspec) distributed checkpoints require"
            " `torch.distributed.checkpoint._fsspec_filesystem`, which is not available in this"
            " PyTorch build. Use a local checkpoint path or upgrade PyTorch."
        ) from e
    return getattr(module, name)
