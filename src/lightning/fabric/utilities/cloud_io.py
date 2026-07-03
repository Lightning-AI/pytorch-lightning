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

import errno
import importlib
import io
import logging
import shutil
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
    if str(path_or_url).startswith("http"):
        if weights_only is None:
            weights_only = False
            log.debug(
                f"Defaulting to `weights_only=False` for remote checkpoint: {path_or_url}."
                f" If loading a checkpoint from an untrustted source, we recommend using `weights_only=True`."
            )

        return torch.hub.load_state_dict_from_url(
            str(path_or_url),
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
        )
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
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
    bytesbuffer = io.BytesIO()
    log.debug(f"Saving checkpoint: {filepath}")
    torch.save(checkpoint, bytesbuffer)

    try:
        # We use a transaction here to avoid file corruption if the save gets interrupted
        fs, urlpath = fsspec.core.url_to_fs(str(filepath))
        with fs.transaction:
            is_azure = False
            if module_available("adlfs"):
                from adlfs import AzureBlobFileSystem

                is_azure = isinstance(fs, AzureBlobFileSystem)

            if _is_object_storage(fs) and not is_azure:
                # Use fs.pipe() for S3/GCS where it triggers parallel multipart uploads,
                # giving 4-5x throughput improvement for checkpoints >= 500 MB.
                # Azure is excluded because adlfs stages blocks sequentially, making pipe() slower.
                fs.pipe(urlpath, bytesbuffer.getvalue())
            else:
                with fs.open(urlpath, "wb") as f:
                    f.write(bytesbuffer.getvalue())
    except PermissionError as e:
        if isinstance(e.__context__, OSError) and getattr(e.__context__, "errno", None) == errno.EXDEV:
            raise RuntimeError(
                'Upgrade fsspec to enable cross-device local checkpoints: pip install "fsspec[http]>=2025.5.0"',
            ) from e


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
        return Path(urlpath)
    return str(path)


def _checkpoint_join(path: Union[str, Path], name: str) -> Union[str, Path]:
    """Join ``name`` onto a checkpoint ``path`` without corrupting remote URLs."""
    if isinstance(path, Path):
        return path / name
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
