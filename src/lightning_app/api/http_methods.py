import asyncio
import inspect
import time
from copy import deepcopy
from functools import wraps
from typing import Callable
from uuid import uuid4

from lightning_app.api.request_types import APIRequest, CommandRequest


def _signature_proxy_function():
    pass


class _HttpMethod:
    def __init__(self, route: str, method: Callable, timeout: int = 15, **kwargs):
        """This class is used to inject user defined methods within the App Rest API.

        Arguments:
            route: The path used to route the requests
            method: The associated flow method
            timeout: The time taken before raising a timeout exception.
        """
        self.route = route
        self.component_name = method.__self__.name
        self.method_name = method.__name__
        self.method_annotations = method.__annotations__
        # TODO: Validate the signature contains only pydantic models.
        self.method_signature = inspect.signature(method)
        self.timeout = timeout
        self.kwargs = kwargs
        self.request_queue = None
        self.response_queue = None

    def add_route(self, app, request_queue, commands_response_store):
        # 1: Create a proxy function with the signature of the wrapped method.
        fn = deepcopy(_signature_proxy_function)
        fn.__annotations__ = self.method_annotations
        fn.__name__ = self.method_name
        setattr(fn, "__signature__", self.method_signature)

        # 2: Get the route associated with the http method.
        route = getattr(app, self.__class__.__name__.lower())

        request_cls = CommandRequest if "command" in self.route else APIRequest

        # 3: Define the request handler.
        @wraps(_signature_proxy_function)
        async def _handle_request(*args, **kwargs):
            async def fn(*args, **kwargs):
                request_id = str(uuid4()).split("-")[0]
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
                while request_id not in commands_response_store:
                    await asyncio.sleep(0.1)
                    if (time.time() - t0) > self.timeout:
                        raise Exception("The response was never received.")

                return commands_response_store.pop(request_id)

            return await asyncio.create_task(fn(*args, **kwargs))

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
