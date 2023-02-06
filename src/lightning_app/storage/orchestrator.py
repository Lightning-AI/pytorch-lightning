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
import threading
import traceback
from queue import Empty
from threading import Thread
from typing import Dict, Optional, TYPE_CHECKING, Union

from lightning_app.core.queues import BaseQueue
from lightning_app.storage.path import _filesystem, _path_to_work_artifact
from lightning_app.storage.requests import _ExistsRequest, _ExistsResponse, _GetRequest, _GetResponse
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.enum import WorkStageStatus

if TYPE_CHECKING:
    from lightning_app import LightningApp


_PathRequest = Union[_GetRequest, _ExistsRequest]
_PathResponse = Union[_ExistsResponse, _GetResponse]
_logger = Logger(__name__)


class StorageOrchestrator(Thread):
    """The StorageOrchestrator processes file transfer requests from Work that need file(s) from other Work.

    Args:
        app: A reference to the ``LightningApp`` which holds the copy request- and response queues for storage.
        request_queues: A dictionary with Queues connected to consumer Work. The Queue will contain transfer requests
            coming from a consumer Work.
        response_queues: A dictionary with Queues connected to consumer Work.
            The Queue will contain the confirmation responses to the consumer Work that files were transferred.
        copy_request_queues: A dictionary of Queues where each Queue connects to one Work. The orchestrator will
            put requests on this queue for the file-transfer thread to complete.
        copy_response_queues: A dictionary of Queues where each Queue connects to one Work. The queue is expected to
            contain the completion response from the file-transfer thread running in the Work process.
    """

    def __init__(
        self,
        app: "LightningApp",
        request_queues: Dict[str, BaseQueue],
        response_queues: Dict[str, BaseQueue],
        copy_request_queues: Dict[str, BaseQueue],
        copy_response_queues: Dict[str, BaseQueue],
    ) -> None:
        super().__init__(daemon=True)
        self.app = app
        self.request_queues = request_queues
        self.response_queues = response_queues
        self.copy_request_queues = copy_request_queues
        self.copy_response_queues = copy_response_queues
        self.waiting_for_response: Dict[str, str] = {}
        self._validate_queues()
        self._exit_event = threading.Event()

        # Note: Use different sleep time locally and in the cloud
        # to reduce queue calls.
        self._sleep_time = 0.1 if "LIGHTNING_APP_STATE_URL" not in os.environ else 2
        self.fs = _filesystem()

    def _validate_queues(self):
        assert (
            self.request_queues.keys()
            == self.response_queues.keys()
            == self.copy_request_queues.keys()
            == self.copy_response_queues.keys()
        )

    def run(self) -> None:
        while not self._exit_event.is_set():
            for work_name in list(self.request_queues.keys()):
                try:
                    self.run_once(work_name)
                except Exception:
                    _logger.error(traceback.format_exc())
            self._exit_event.wait(self._sleep_time)

    def join(self, timeout: Optional[float] = None) -> None:
        self._exit_event.set()
        super().join(timeout)

    def run_once(self, work_name: str) -> None:
        if work_name not in self.waiting_for_response:
            # check if there is a new request from this work for a file transfer
            # there can only be one pending request per work
            request_queue = self.request_queues[work_name]
            try:
                request: _PathRequest = request_queue.get(timeout=0)  # this should not block
            except Empty:
                pass
            else:
                request.destination = work_name
                source_work = self.app.get_component_by_name(request.source)
                maybe_artifact_path = str(_path_to_work_artifact(request.path, source_work))

                if self.fs.exists(maybe_artifact_path):
                    # First check if the shared filesystem has the requested file stored as an artifact
                    # If so, we will let the destination Work access this file directly
                    # NOTE: This is NOT the right thing to do, because the Work could still be running and producing
                    # a newer version of the requested file, but we can't rely on the Work status to be accurate
                    # (at the moment)
                    if isinstance(request, _GetRequest):
                        response = _GetResponse(
                            source=request.source,
                            name=request.name,
                            path=maybe_artifact_path,
                            hash=request.hash,
                            size=self.fs.info(maybe_artifact_path)["size"],
                            destination=request.destination,
                        )
                    if isinstance(request, _ExistsRequest):
                        response = _ExistsResponse(
                            source=request.source,
                            path=maybe_artifact_path,
                            name=request.name,
                            hash=request.hash,
                            destination=request.destination,
                            exists=True,
                        )
                    response_queue = self.response_queues[response.destination]
                    response_queue.put(response)
                elif source_work.status.stage not in (
                    WorkStageStatus.NOT_STARTED,
                    WorkStageStatus.STOPPED,
                    WorkStageStatus.FAILED,
                ):
                    _logger.debug(
                        f"Request for File Transfer received from {work_name}: {request}. Sending request to"
                        f" {request.source} to copy the file."
                    )
                    # The Work is running, and we can send a request to the copier for moving the file to the
                    # shared storage
                    self.copy_request_queues[request.source].put(request)
                    # Store a destination to source mapping.
                    self.waiting_for_response[work_name] = request.source
                else:
                    if isinstance(request, _GetRequest):
                        response = _GetResponse(
                            source=request.source,
                            path=request.path,
                            name=request.name,
                            hash=request.hash,
                            size=0,
                            destination=request.destination,
                        )
                    if isinstance(request, _ExistsRequest):
                        response = _ExistsResponse(
                            source=request.source,
                            path=request.path,
                            hash=request.hash,
                            destination=request.destination,
                            exists=False,
                            name=request.name,
                        )
                    response.exception = FileNotFoundError(
                        "The work is not running and the requested object is not available in the artifact store."
                    )
                    response_queue = self.response_queues[response.destination]
                    response_queue.put(response)

        # Check the current work is within the sources.
        # It is possible to have multiple destination targeting
        # the same source concurrently.
        if work_name in self.waiting_for_response.values():

            # check if the current work has responses for file transfers to other works.
            copy_response_queue = self.copy_response_queues[work_name]
            try:
                # check if the share-point file manager has confirmed a copy request
                response: _PathResponse = copy_response_queue.get(timeout=0)  # this should not block
            except Empty:
                pass
            else:
                _logger.debug(
                    f"Received confirmation of a completed file copy request from {work_name}:{response}."
                    f" Sending the confirmation back to {response.destination}."
                )
                destination = response.destination
                assert response.source == work_name
                response_queue = self.response_queues[destination]
                response_queue.put(response)
                # the request has been processed, allow new requests to come in for the destination work
                del self.waiting_for_response[destination]
