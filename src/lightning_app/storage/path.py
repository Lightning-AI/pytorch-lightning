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

import hashlib
import os
import pathlib
import shutil
import sys
from time import sleep
from typing import Any, List, Optional, Sequence, TYPE_CHECKING, Union

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from lightning_app.core.constants import REMOTE_STORAGE_WAIT
from lightning_app.core.queues import BaseQueue
from lightning_app.storage.requests import _ExistsRequest, _ExistsResponse, _GetRequest, _GetResponse
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.component import _is_flow_context
from lightning_app.utilities.imports import _is_s3fs_available

if _is_s3fs_available():
    from s3fs import S3FileSystem

PathlibPath = type(pathlib.Path())  # PosixPath or a WindowsPath depending on the platform

if TYPE_CHECKING:
    from lightning_app.core.work import LightningWork

num_workers = 8

_logger = Logger(__name__)


class Path(PathlibPath):
    """A drop-in replacement for :class:`pathlib.Path` for all paths in Lightning.

    The Lightning Path works exactly the same as :class:`pathlib.Path` but it also remembers in which LightningWork
    it was created. If the Path gets passed to a different LightningWork, the file or folder can then be easily
    accessed no matter where it is located in the other Work's filesystem.

    Args:
        *args: Accepts the same arguments as in :class:`pathlib.Path`
        **kwargs: Accepts the same keyword arguments as in :class:`pathlib.Path`
    """

    @classmethod
    def _from_parts(cls, args: Any, **__unused) -> "Path":
        """This gets called from the super class in ``pathlib.Path.__new__``.

        The Lightning Path overrides this to validate the instantiation in the case parts are passed in individually. In
        such a case we need to validate that all parts have the same `origin` and if not, an error is raised.
        """
        if args and isinstance(args[0], str) and args[0].startswith("lit://"):
            parts = list(args)
            parts[0] = parts[0][len("lit://") :]
            args = (_storage_root_dir(), *parts)

        if (sys.version_info.major, sys.version_info.minor) < (3, 10):
            __unused.setdefault("init", True)
            new_path = super()._from_parts(args, **__unused)
        else:
            new_path = super()._from_parts(args)

        new_path._init_attributes()  # we use this instead of defining a __init__() method

        paths_from_parts = [part for part in args if isinstance(part, Path)]
        if not paths_from_parts:
            return new_path
        top_path = paths_from_parts[0]
        origins = [part._origin for part in paths_from_parts]
        if not all(origins[0] == origin or origin is None for origin in origins):
            raise TypeError(
                "Tried to instantiate a Lightning Path from multiple other Paths that originate from different"
                " LightningWork."
            )
        new_path._copy_properties_from(top_path)
        return new_path

    def _init_attributes(self):
        self._name: Optional[str] = None
        # the origin is the work that created this Path and wants to expose file(s)
        self._origin: Optional[Union["LightningWork", str]] = None
        # the consumer is the Work that needs access to the file(s) from the consumer
        self._consumer: Optional[Union["LightningWork", str]] = None
        self._metadata = {}
        # request queue: used to transfer message to storage orchestrator
        self._request_queue: Optional[BaseQueue] = None
        # response queue: used to receive status message from storage orchestrator
        self._response_queue: Optional[BaseQueue] = None

    @property
    def origin_name(self) -> str:
        """The name of the LightningWork where this path was first created.

        Attaching a Path to a LightningWork will automatically make it the `origin`.
        """
        from lightning_app.core.work import LightningWork

        return self._origin.name if isinstance(self._origin, LightningWork) else self._origin

    @property
    def consumer_name(self) -> str:
        """The name of the LightningWork where this path is being accessed.

        By default, this is the same as the :attr:`origin_name`.
        """
        from lightning_app.core.work import LightningWork

        return self._consumer.name if isinstance(self._consumer, LightningWork) else self._consumer

    @property
    def hash(self) -> Optional[str]:
        """The hash of this Path uniquely identifies the file path and the associated origin Work.

        Returns ``None`` if the origin is not defined, i.e., this Path did not yet get attached to a LightningWork.
        """
        if self._origin is None:
            return None
        contents = f"{self.origin_name}/{self}"
        return hashlib.sha1(contents.encode("utf-8")).hexdigest()

    @property
    def parents(self) -> Sequence["Path"]:
        parents: List["Path"] = list(super().parents)
        for parent in parents:
            parent._copy_properties_from(self)
        return parents

    @property
    def parent(self) -> "Path":
        parent: Path = super().parent
        parent._copy_properties_from(self)
        return parent

    def exists(self) -> bool:
        """Check if the path exists locally or remotely.

        If the path exists locally, this method immediately returns ``True``, otherwise it will make a RPC call
        to the attached origin Work and check if the path exists remotely.
        If you strictly want to check local existence only, use :meth:`exists_local` instead. If you strictly want
        to check existence on the remote (regardless of whether the file exists locally or not), use
        :meth:`exists_remote`.
        """
        return self.exists_local() or (self._origin and self.exists_remote())

    def exists_local(self) -> bool:
        """Check if the path exists locally."""
        return super().exists()

    def exists_remote(self) -> bool:
        """Check if the path exists remotely on the attached orgin Work.

        Raises:
            RuntimeError: If the path is not attached to any Work (origin undefined).
        """
        # Fail early if we need to check the remote but an origin is not defined
        if not self._origin or self._request_queue is None or self._response_queue is None:
            raise RuntimeError(
                f"Trying to check if the file {self} exists, but the path is not attached to a LightningWork."
                f" Set it as an attribute to a LightningWork or pass it to the `run()` method."
            )

        # 1. Send message to orchestrator through queue that with a request for a path existence check
        request = _ExistsRequest(source=self.origin_name, path=str(self), name=self._name, hash=self.hash)
        self._request_queue.put(request)

        # 2. Wait for the response to come back
        response: _ExistsResponse = self._response_queue.get()  # blocking
        return response.exists

    def get(self, overwrite: bool = False) -> None:
        if _is_flow_context():
            raise RuntimeError("`Path.get()` can only be called from within the `run()` method of LightningWork.")
        if self._request_queue is None or self._response_queue is None:
            raise RuntimeError(
                f"Trying to get the file {self}, but the path is not attached to a LightningApp."
                f" Are you trying to get the file from within `__init__`?"
            )
        if self._origin is None:
            raise RuntimeError(
                f"Trying to get the file {self}, but the path is not attached to a LightningWork. Set it as an"
                f" attribute to a LightningWork or pass it to the `run()` method."
            )

        if self.exists_local() and not overwrite:
            raise FileExistsError(
                f"The file or folder {self} exists locally. Pass `overwrite=True` if you wish to replace it"
                f" with the new contents."
            )

        # 1. Send message to orchestrator through queue with details of the transfer
        # the source is the name of the work that owns the file that we request
        # the destination is determined by the queue, since each work has a dedicated send and recv queue
        request = _GetRequest(source=self.origin_name, path=str(self), hash=self.hash, name=self._name)
        self._request_queue.put(request)

        # 2. Wait for the transfer to finish
        response: _GetResponse = self._response_queue.get()  # blocking
        self._validate_get_response(response)

        fs = _filesystem()

        # 3. Wait until the file appears in shared storage
        while not fs.exists(response.path) or fs.info(response.path)["size"] != response.size:
            sleep(REMOTE_STORAGE_WAIT)

        if self.exists_local() and self.is_dir():
            # Delete the directory, otherwise we can't overwrite it
            shutil.rmtree(self)

        # 4. Copy the file from the shared storage to the destination on the local filesystem
        if fs.isdir(response.path):
            if isinstance(fs, LocalFileSystem):
                shutil.copytree(response.path, self.resolve())
            else:
                glob = f"{str(response.path)}/**"
                _logger.debug(f"Attempting to copy {glob} -> {str(self.absolute())}")
                fs.get(glob, str(self.absolute()), recursive=False)
        else:
            _logger.debug(f"Attempting to copy {str(response.path)} -> {str(self.absolute())}")
            fs.get(str(response.path), str(self.absolute()), recursive=False)

    def to_dict(self) -> dict:
        """Serialize this Path to a dictionary."""
        return dict(
            path=str(self),
            origin_name=self.origin_name,
            consumer_name=self.consumer_name,
            metadata=self._metadata,
        )

    @classmethod
    def from_dict(cls, content: dict) -> "Path":
        """Instantiate a Path from a dictionary."""
        path = cls(content["path"])
        path._origin = content["origin_name"]
        path._consumer = content["consumer_name"]
        path._metadata = content["metadata"]
        return path

    def _validate_get_response(self, response: "_GetResponse") -> None:
        if response.source != self._origin or response.hash != self.hash:
            raise RuntimeError(
                f"Tried to get the file {self} but received a response for a request it did not send. The response"
                f" contents are: {response}"
            )

        if response.exception is not None:
            raise RuntimeError(
                f"An exception was raised while trying to transfer the contents at {response.path}"
                f" from Work {response.source} to {response.destination}. See the full stacktrace above."
            ) from response.exception

    def _attach_work(self, work: "LightningWork") -> None:
        """Attach a LightningWork to this Path.

        The first work to be attached becomes the `origin`, i.e., the Work that is meant to expose the file to other
        Work. Attaching a Work to a Path that already has an `origin` Work will make it a `consumer`. A consumer Work
        is a work that can access the file only by first transferring it via :meth:`transfer`.

        Args:
            work: LightningWork to be attached to this Path.
        """
        if self._origin is None:
            # Can become an owner only if there is not already one
            self._origin = work
        self._consumer = work

    def _attach_queues(self, request_queue: BaseQueue, response_queue: BaseQueue) -> None:
        """Attaches the queues for communication with the Storage Orchestrator."""
        self._request_queue = request_queue
        self._response_queue = response_queue

    def _sanitize(self) -> None:
        """Sanitize this Path so that it can be deep-copied."""
        self._origin = self.origin_name
        self._consumer = self.consumer_name
        self._request_queue = None
        self._response_queue = None

    def _copy_properties_from(self, other: "Path") -> None:
        self._origin = other._origin
        self._consumer = other._consumer
        self._metadata = other._metadata
        self._request_queue = other._request_queue
        self._response_queue = other._response_queue

    def with_name(self, name: str) -> "Path":
        path: Path = super().with_name(name)
        path._copy_properties_from(self)
        return path

    def with_stem(self, stem: str) -> "Path":
        path: Path = super().with_stem(stem)
        path._copy_properties_from(self)
        return path

    def with_suffix(self, suffix: str) -> "Path":
        path: Path = super().with_suffix(suffix)
        path._copy_properties_from(self)
        return path

    def relative_to(self, *other) -> "Path":
        path: Path = super().relative_to(*other)
        path._copy_properties_from(self)
        return path

    def __truediv__(self, other: Union["Path", PathlibPath, str]) -> "Path":
        path: Path = super().__truediv__(other)
        path._copy_properties_from(self)
        return path

    def __rtruediv__(self, other: Union["Path", PathlibPath, str]) -> "Path":
        path: Path = super().__rtruediv__(other)
        path._copy_properties_from(self)
        return path

    def __reduce__(self):
        return Path.from_dict, (self.to_dict(),)

    def __json__(self) -> dict:
        """Converts the Path to a json-serializable dict object."""
        return self.to_dict()

    @staticmethod
    def _handle_exists_request(work: "LightningWork", request: _ExistsRequest) -> _ExistsResponse:
        return _ExistsResponse(
            source=request.source,
            name=request.name,
            hash=request.hash,
            path=request.path,
            destination=request.destination,
            exists=os.path.exists(request.path),
        )

    @staticmethod
    def _handle_get_request(work: "LightningWork", request: _GetRequest) -> _GetResponse:
        from lightning_app.storage.copier import _copy_files

        source_path = pathlib.Path(request.path)
        destination_path = _shared_storage_path() / request.hash
        response = _GetResponse(
            source=request.source,
            name=request.name,
            path=str(destination_path),
            hash=request.hash,
            size=source_path.stat().st_size,
            destination=request.destination,
        )

        try:
            _copy_files(source_path, destination_path)
            _logger.debug(f"All files copied from {request.path} to {response.path}.")
        except Exception as e:
            response.exception = e
        return response


def _is_lit_path(path: Union[str, Path]) -> bool:
    path = Path(path)
    return path == _storage_root_dir() or _storage_root_dir() in path.parents


def _shared_local_mount_path() -> pathlib.Path:
    """Returns the shared directory through which the Copier threads move files from one Work filesystem to
    another.

    The shared directory can be set via the environment variable ``SHARED_MOUNT_DIRECTORY`` and should be pointing to a
    directory that all Works have mounted (shared filesystem).
    """
    path = pathlib.Path(os.environ.get("SHARED_MOUNT_DIRECTORY", ".shared"))
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute()


def _storage_root_dir() -> pathlib.Path:
    path = pathlib.Path(os.environ.get("STORAGE_ROOT_DIR", "./.storage")).absolute()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _shared_storage_path() -> pathlib.Path:
    """Returns the shared path through which the Copier threads move files from one Work filesystem to another.

    The shared path gets set by the environment. Locally, it is pointing to a directory determined by the
    ``SHARED_MOUNT_DIRECTORY`` environment variable. In the cloud, the shared path will point to a S3 bucket. All Works
    have access to this shared dropbox.
    """
    storage_path = os.getenv("LIGHTNING_STORAGE_PATH", "")
    if storage_path != "":
        return pathlib.Path(storage_path)

    # TODO[dmitsf]: this logic is still needed for compatibility reasons.
    # We should remove it after some time.
    bucket_name = os.getenv("LIGHTNING_BUCKET_NAME", "")
    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", "")

    if bucket_name != "" and app_id != "":
        return pathlib.Path(f"{bucket_name}/lightningapps/{app_id}")

    return _shared_local_mount_path()


def _artifacts_path(work: "LightningWork") -> pathlib.Path:
    return _shared_storage_path() / "artifacts" / work.name


def _path_to_work_artifact(path: Union[Path, pathlib.Path, str], work: "LightningWork") -> pathlib.Path:
    return _artifacts_path(work) / pathlib.Path(*pathlib.Path(path).absolute().parts[1:])


def _filesystem() -> AbstractFileSystem:
    fs = LocalFileSystem()

    endpoint_url = os.getenv("LIGHTNING_BUCKET_ENDPOINT_URL", "")
    bucket_name = os.getenv("LIGHTNING_BUCKET_NAME", "")
    if endpoint_url != "" and bucket_name != "":
        key = os.getenv("LIGHTNING_AWS_ACCESS_KEY_ID", "")
        secret = os.getenv("LIGHTNING_AWS_SECRET_ACCESS_KEY", "")
        # TODO: Remove when updated on the platform side.
        if key == "" or secret == "":
            key = os.getenv("AWS_ACCESS_KEY_ID", "")
            secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        if key == "" or secret == "":
            raise RuntimeError("missing S3 bucket credentials")

        fs = S3FileSystem(key=key, secret=secret, use_ssl=False, client_kwargs={"endpoint_url": endpoint_url})

        app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", "")
        if app_id == "":
            raise RuntimeError("missing LIGHTNING_CLOUD_APP_ID")

        if not fs.exists(_shared_storage_path()):
            raise RuntimeError(f"shared filesystem {_shared_storage_path()} does not exist")

    return fs
