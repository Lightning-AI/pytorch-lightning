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
import pathlib
import pickle
from abc import ABC, abstractmethod
from time import sleep
from typing import Any, Optional, TYPE_CHECKING, Union

from lightning.app.core.constants import REMOTE_STORAGE_WAIT
from lightning.app.core.queues import BaseQueue
from lightning.app.storage.path import _filesystem, _shared_storage_path, Path
from lightning.app.storage.requests import _ExistsRequest, _ExistsResponse, _GetRequest, _GetResponse
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.component import _is_flow_context

_logger = Logger(__name__)

if TYPE_CHECKING:
    from lightning.app.core.work import LightningWork


class _BasePayload(ABC):
    def __init__(self, value: Any) -> None:
        self._value = value
        # the attribute name given to the payload
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
    def name(self) -> Optional[str]:
        return self._name

    @property
    def value(self) -> Optional[Any]:
        """The real object that this payload holds."""
        return self._value

    @property
    def hash(self) -> Optional[str]:
        """The hash of this Payload uniquely identifies the payload and the associated origin Work.

        Returns ``None`` if the origin is not defined, i.e., this Path did not yet get attached to a LightningWork.
        """
        if self._origin is None:
            return None
        contents = f"{self.origin_name}/{self.consumer_name}/{self.name}"
        return hashlib.sha1(contents.encode("utf-8")).hexdigest()

    @property
    def origin_name(self) -> str:
        """The name of the LightningWork where this payload was first created.

        Attaching a Payload to a LightningWork will automatically make it the `origin`.
        """
        from lightning.app.core.work import LightningWork

        return self._origin.name if isinstance(self._origin, LightningWork) else self._origin

    @property
    def consumer_name(self) -> str:
        """The name of the LightningWork where this payload is being accessed.

        By default, this is the same as the :attr:`origin_name`.
        """
        from lightning.app.core.work import LightningWork

        return self._consumer.name if isinstance(self._consumer, LightningWork) else self._consumer

    @property
    def _path(self) -> Optional[Path]:
        """Path to the file that the payload value gets serialized to."""
        if not self._name:
            return
        return Path("lit://", self._name)

    @abstractmethod
    def save(self, obj: Any, path: str) -> None:
        """Override this method with your own saving logic."""

    @abstractmethod
    def load(self, path: str) -> Any:
        """Override this method with your own loading logic."""

    def _attach_work(self, work: "LightningWork") -> None:
        """Attach a LightningWork to this PayLoad.

        Args:
            work: LightningWork to be attached to this Payload.
        """
        if self._origin is None:
            # Can become an owner only if there is not already one
            self._origin = work.name
        self._consumer = work.name

    def _attach_queues(self, request_queue: BaseQueue, response_queue: BaseQueue) -> None:
        """Attaches the queues for communication with the Storage Orchestrator."""
        self._request_queue = request_queue
        self._response_queue = response_queue

    def _sanitize(self) -> None:
        """Sanitize this Payload so that it can be deep-copied."""
        self._origin = self.origin_name
        self._consumer = self.consumer_name
        self._request_queue = None
        self._response_queue = None

    def exists_remote(self):
        """Check if the payload exists remotely on the attached orgin Work.

        Raises:
            RuntimeError: If the payload is not attached to any Work (origin undefined).
        """
        # Fail early if we need to check the remote but an origin is not defined
        if not self._origin or self._request_queue is None or self._response_queue is None:
            raise RuntimeError(
                f"Trying to check if the payload {self} exists, but the payload is not attached to a LightningWork."
                f" Set it as an attribute to a LightningWork or pass it to the `run()` method."
            )

        # 1. Send message to orchestrator through queue that with a request for a path existence check
        request = _ExistsRequest(source=self.origin_name, name=self._name, path=str(self._path), hash=self.hash)
        self._request_queue.put(request)

        # 2. Wait for the response to come back
        response: _ExistsResponse = self._response_queue.get()  # blocking
        return response.exists

    def get(self) -> Any:
        if _is_flow_context():
            raise RuntimeError("`Payload.get()` can only be called from within the `run()` method of LightningWork.")

        if self._request_queue is None or self._response_queue is None:
            raise RuntimeError(
                f"Trying to get the file {self}, but the payload is not attached to a LightningApp."
                f" Are you trying to get the file from within `__init__`?"
            )
        if self._origin is None:
            raise RuntimeError(
                f"Trying to get the file {self}, but the payload is not attached to a LightningWork. Set it as an"
                f" attribute to a LightningWork or pass it to the `run()` method."
            )

        # 1. Send message to orchestrator through queue with details of the transfer
        # the source is the name of the work that owns the file that we request
        # the destination is determined by the queue, since each work has a dedicated send and recv queue
        request = _GetRequest(source=self.origin_name, name=self._name, path=str(self._path), hash=self.hash)
        self._request_queue.put(request)

        # 2. Wait for the transfer to finish
        response: _GetResponse = self._response_queue.get()  # blocking
        self._validate_get_response(response)

        fs = _filesystem()

        # 3. Wait until the file appears in shared storage
        while not fs.exists(response.path) or fs.info(response.path)["size"] != response.size:
            sleep(REMOTE_STORAGE_WAIT)

        # 4. Copy the file from the shared storage to the destination on the local filesystem
        local_path = self._path
        _logger.debug(f"Attempting to copy {str(response.path)} -> {str(local_path)}")
        fs.get(str(response.path), str(local_path), recursive=False)

        # Ensure the file is properly written
        sleep(0.5)

        self._value = self.load(local_path)
        return self._value

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

    def to_dict(self) -> dict:
        """Serialize this Path to a dictionary."""
        return dict(
            name=self.name,
            origin_name=self.origin_name,
            consumer_name=self.consumer_name,
            metadata=self._metadata,
        )

    @classmethod
    def from_dict(cls, content: dict) -> "_BasePayload":
        """Instantiate a Payload from a dictionary."""
        payload = cls(None)
        payload._name = content["name"]
        payload._origin = content["origin_name"]
        payload._consumer = content["consumer_name"]
        payload._metadata = content["metadata"]
        return payload

    @staticmethod
    def _handle_exists_request(work: "LightningWork", request: _ExistsRequest) -> _ExistsResponse:
        return _ExistsResponse(
            source=request.source,
            path=request.path,
            name=request.name,
            destination=request.destination,
            hash=request.hash,
            exists=getattr(work, request.name, None) is not None,
        )

    @staticmethod
    def _handle_get_request(work: "LightningWork", request: _GetRequest) -> _GetResponse:
        from lightning.app.storage.copier import _copy_files

        source_path = pathlib.Path(request.path)
        destination_path = _shared_storage_path() / request.hash
        response = _GetResponse(
            source=request.source,
            name=request.name,
            path=str(destination_path),
            hash=request.hash,
            destination=request.destination,
        )

        try:
            payload = getattr(work, request.name)
            payload.save(payload.value, source_path)
            response.size = source_path.stat().st_size
            _copy_files(source_path, destination_path)
            _logger.debug(f"All files copied from {request.path} to {response.path}.")
        except Exception as e:
            response.exception = e
        return response


class Payload(_BasePayload):
    """The Payload object enables to transfer python objects from one work to another in a similar fashion as
    :class:`~lightning.app.storage.path.Path`."""

    def save(self, obj: Any, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load(self, path: str) -> Any:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
