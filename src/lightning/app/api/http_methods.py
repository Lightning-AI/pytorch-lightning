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

import asyncio
import inspect
import time
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from multiprocessing import Queue
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from lightning_utilities.core.apply_func import apply_to_collection

from lightning.app.api.request_types import _APIRequest, _CommandRequest, _RequestResponse
from lightning.app.utilities.app_helpers import Logger

logger = Logger(__name__)


def _signature_proxy_function():
    pass


@dataclass
class _FastApiMockRequest:
    """This class is meant to mock FastAPI Request class that isn't pickle-able.

    If a user relies on FastAPI Request annotation, the Lightning framework
    patches the annotation before pickling and replace them right after.

    Finally, the FastAPI request is converted back to the _FastApiMockRequest
    before being delivered to the users.

    Example:

        import lightning as L
        from fastapi import Request
        from lightning.app.api import Post

        class Flow(L.LightningFlow):

            def request(self, request: Request) -> OutputRequestModel:
                ...

            def configure_api(self):
                return [Post("/api/v1/request", self.request)]
    """

    _body: Optional[str] = None
    _json: Optional[str] = None
    _method: Optional[str] = None
    _headers: Optional[Dict] = None

    @property
    def receive(self):
        raise NotImplementedError

    @property
    def method(self):
        raise self._method

    @property
    def headers(self):
        return self._headers

    def body(self):
        return self._body

    def json(self):
        return self._json

    def stream(self):
        raise NotImplementedError

    def form(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def is_disconnected(self):
        raise NotImplementedError


async def _mock_fastapi_request(request: Request):
    # TODO: Add more requests parameters.
    return _FastApiMockRequest(
        _body=await request.body(),
        _json=await request.json(),
        _headers=request.headers,
        _method=request.method,
    )


class _HttpMethod:
    def __init__(self, route: str, method: Callable, method_name: Optional[str] = None, timeout: int = 30, **kwargs):
        """This class is used to inject user defined methods within the App Rest API.

        Arguments:
            route: The path used to route the requests
            method: The associated flow method
            timeout: The time in seconds taken before raising a timeout exception.
        """
        self.route = route
        self.attached_to_flow = hasattr(method, "__self__")
        self.method_name = method_name or method.__name__
        self.method_annotations = method.__annotations__
        # TODO: Validate the signature contains only pydantic models.
        self.method_signature = inspect.signature(method)

        if not self.attached_to_flow:
            self.component_name = method.__name__
            self.method = method
        else:
            self.component_name = method.__self__.name

        self.timeout = timeout
        self.kwargs = kwargs

        # Enable the users to rely on FastAPI annotation typing with Request.
        # Note: Only a part of the Request functionatilities are supported.
        self._patch_fast_api_request()

    def add_route(self, app: FastAPI, request_queue: Queue, responses_store: Dict[str, Any]) -> None:
        # 1: Get the route associated with the http method.
        route = getattr(app, self.__class__.__name__.lower())

        self._unpatch_fast_api_request()

        # 2: Create a proxy function with the signature of the wrapped method.
        fn = deepcopy(_signature_proxy_function)
        fn.__annotations__ = self.method_annotations
        fn.__name__ = self.method_name
        setattr(fn, "__signature__", self.method_signature)

        # Note: Handle requests differently if attached to a flow.
        if not self.attached_to_flow:
            # 3: Define the request handler.
            @wraps(_signature_proxy_function)
            async def _handle_request(*args, **kwargs):
                if inspect.iscoroutinefunction(self.method):
                    return await self.method(*args, **kwargs)
                return self.method(*args, **kwargs)

        else:
            request_cls = _CommandRequest if self.route.startswith("/command/") else _APIRequest

            # 3: Define the request handler.
            @wraps(_signature_proxy_function)
            async def _handle_request(*args, **kwargs):
                async def fn(*args, **kwargs):
                    args, kwargs = apply_to_collection((args, kwargs), Request, _mock_fastapi_request)
                    for k, v in kwargs.items():
                        if hasattr(v, "__await__"):
                            kwargs[k] = await v

                    request_id = str(uuid4()).split("-")[0]
                    logger.debug(f"Processing request {request_id} for route: {self.route}")
                    request_queue.put(
                        request_cls(
                            name=self.component_name,
                            method_name=self.method_name,
                            args=args,
                            kwargs=kwargs,
                            id=request_id,
                        )
                    )

                    t0 = time.time()
                    while request_id not in responses_store:
                        await asyncio.sleep(0.01)
                        if (time.time() - t0) > self.timeout:
                            raise HTTPException(
                                status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="The response was never received.",
                            )

                    logger.debug(f"Processed request {request_id} for route: {self.route}")

                    return responses_store.pop(request_id)

                response: _RequestResponse = await asyncio.create_task(fn(*args, **kwargs))

                if response.status_code != 200:
                    raise HTTPException(response.status_code, detail=response.content)

                return response.content

        # 4: Register the user provided route to the Rest API.
        route(self.route, **self.kwargs)(_handle_request)

    def _patch_fast_api_request(self):
        """This function replaces signature annotation for Request with its mock."""
        for k, v in self.method_annotations.items():
            if v == Request:
                self.method_annotations[k] = _FastApiMockRequest

        for v in self.method_signature.parameters.values():
            if v._annotation == Request:
                v._annotation = _FastApiMockRequest

    def _unpatch_fast_api_request(self):
        """This function replaces back signature annotation to fastapi Request."""
        for k, v in self.method_annotations.items():
            if v == _FastApiMockRequest:
                self.method_annotations[k] = Request

        for v in self.method_signature.parameters.values():
            if v._annotation == _FastApiMockRequest:
                v._annotation = Request


class Post(_HttpMethod):
    pass


class Get(_HttpMethod):
    pass


class Put(_HttpMethod):
    pass


class Delete(_HttpMethod):
    pass


def _add_tags_to_api(apis: List[_HttpMethod], tags: List[str]) -> None:
    for api in apis:
        if not api.kwargs.get("tag"):
            api.kwargs["tags"] = tags


def _validate_api(apis: List[_HttpMethod]) -> None:
    for api in apis:
        if not isinstance(api, _HttpMethod):
            raise Exception(f"The provided api should be either [{Delete}, {Get}, {Post}, {Put}]")
        if api.route.startswith("/command"):
            raise Exception("The route `/command` is reserved for commands. Please, use something else.")
