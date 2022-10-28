import asyncio
import inspect
import time
from copy import deepcopy
from functools import wraps
from multiprocessing import Queue
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from lightning_app.api.request_types import _APIRequest, _CommandRequest, _RequestResponse
from lightning_app.utilities.app_helpers import Logger

logger = Logger(__name__)


def _signature_proxy_function():
    pass


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

    def add_route(self, app: FastAPI, request_queue: Queue, responses_store: Dict[str, Any]) -> None:
        # 1: Get the route associated with the http method.
        route = getattr(app, self.__class__.__name__.lower())

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
                            raise Exception("The response was never received.")

                    logger.debug(f"Processed request {request_id} for route: {self.route}")

                    return responses_store.pop(request_id)

                response: _RequestResponse = await asyncio.create_task(fn(*args, **kwargs))

                if response.status_code != 200:
                    raise HTTPException(response.status_code, detail=response.content)

                return response.content

        # 4: Register the user provided route to the Rest API.
        route(self.route, **self.kwargs)(_handle_request)


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
