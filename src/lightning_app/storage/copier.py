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

import concurrent.futures
import pathlib
import threading
from threading import Thread
from time import time
from typing import Optional, TYPE_CHECKING, Union

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from lightning_app.core.queues import BaseQueue
from lightning_app.storage.path import _filesystem
from lightning_app.storage.requests import _ExistsRequest, _GetRequest
from lightning_app.utilities.app_helpers import Logger

_PathRequest = Union[_GetRequest, _ExistsRequest]

_logger = Logger(__name__)

num_workers = 8
if TYPE_CHECKING:
    import lightning_app


class _Copier(Thread):
    """The Copier is a thread running alongside a LightningWork.

    It maintains two queues that connect to the central
    :class:`~lightning_app.storage.orchestrator.StorageOrchestrator`,
    the request queue and the response queue. The Copier waits for a request to be pushed to the request queue,
    processes it and sends back the request through the response queue. In the current implementation, the Copier
    simply copies the requested file from the local filesystem to a shared directory (determined by
    :func:`~lightning_app.storage.path.shared_storage_path`). Any errors raised during the copy will be added to the
    response and get re-raised within the corresponding LightningWork.

    Args:
        copy_request_queue: A queue connecting the central StorageOrchestrator with the Copier. The orchestrator
            will send requests to this queue.
        copy_response_queue: A queue connecting the central StorageOrchestrator with the Copier. The Copier
            will send a response to this queue whenever a requested copy has finished.
    """

    def __init__(
        self, work: "lightning_app.LightningWork", copy_request_queue: "BaseQueue", copy_response_queue: "BaseQueue"
    ) -> None:
        super().__init__(daemon=True)
        self._work = work
        self.copy_request_queue = copy_request_queue
        self.copy_response_queue = copy_response_queue
        self._exit_event = threading.Event()
        self._sleep_time = 0.1

    def run(self) -> None:
        while not self._exit_event.is_set():
            self._exit_event.wait(self._sleep_time)
            self.run_once()

    def join(self, timeout: Optional[float] = None) -> None:
        self._exit_event.set()
        super().join(timeout)

    def run_once(self):
        request: _PathRequest = self.copy_request_queue.get()  # blocks until we get a request

        t0 = time()

        obj: Optional[lightning_app.storage.Path] = _find_matching_path(self._work, request)
        if obj is None:
            # If it's not a path, it must be a payload
            obj: lightning_app.storage.Payload = getattr(self._work, request.name)

        if isinstance(request, _ExistsRequest):
            response = obj._handle_exists_request(self._work, request)
        elif isinstance(request, _GetRequest):
            response = obj._handle_get_request(self._work, request)
        else:
            raise TypeError(
                f"The file copy request had an invalid type. Expected PathGetRequest or PathExistsRequest, got:"
                f" {type(request)}"
            )

        response.timedelta = time() - t0
        self.copy_response_queue.put(response)


def _find_matching_path(work, request: _GetRequest) -> Optional["lightning_app.storage.Path"]:
    for name in work._paths:
        candidate: lightning_app.storage.Path = getattr(work, name)
        if candidate.hash == request.hash:
            return candidate


def _copy_files(
    source_path: pathlib.Path,
    destination_path: pathlib.Path,
    fs: Optional[AbstractFileSystem] = None,
) -> None:
    """Copy files from one path to another.

    The source path must either be an existing file or folder. If the source is a folder, the destination path is
    interpreted as a folder as well. If the source is a file, the destination path is interpreted as a file too.

    Files in a folder are copied recursively and efficiently using multiple threads.
    """
    if fs is None:
        fs = _filesystem()

    def _copy(from_path: pathlib.Path, to_path: pathlib.Path) -> Optional[Exception]:
        _logger.debug(f"Copying {str(from_path)} -> {str(to_path)}")

        try:
            # NOTE: S3 does not have a concept of directories, so we do not need to create one.
            if isinstance(fs, LocalFileSystem):
                fs.makedirs(str(to_path.parent), exist_ok=True)

            fs.put(str(from_path), str(to_path), recursive=False)
        except Exception as e:
            # Return the exception so that it can be handled in the main thread
            return e

    # NOTE: Cannot use `S3FileSystem.put(recursive=True)` because it tries to access parent directories
    #       which it does not have access to.
    if source_path.is_dir():
        src = [file for file in source_path.rglob("*") if file.is_file()]
        dst = [destination_path / file.relative_to(source_path) for file in src]

        with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
            results = executor.map(_copy, src, dst)

        # Raise the first exception found
        exception = next((e for e in results if isinstance(e, Exception)), None)
        if exception:
            raise exception
    else:
        if isinstance(fs, LocalFileSystem):
            fs.makedirs(str(destination_path.parent), exist_ok=True)

        fs.put(str(source_path), str(destination_path))
