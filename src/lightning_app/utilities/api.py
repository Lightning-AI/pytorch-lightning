import asyncio
import inspect
import time
from copy import deepcopy
from functools import wraps
from typing import Callable
from uuid import uuid4


def signature_proxy_function():
    pass


class Protocol:

    name = None

    def __init__(self, route: str, method: Callable, timeout: int = 15, **kwargs):
        """This class is used to inject user defined methods within the App Rest API.

        Arguments:
            route: The path used to route the requests
            method: The associated flow method
            timeout: The time taken before raising a timeout exception.
        """
        self.route = route
        self.method = method
        self.flow_name = method.__self__.name
        self.timeout = timeout
        self.kwargs = kwargs
        self.request_queue = None
        self.response_queue = None

    def add_route(self, app, request_queue, commands_response_store):
        assert self.name is not None
        # 1: Create a proxy function with the same signature for FastAPI
        # swagger UI.
        fn = deepcopy(signature_proxy_function)
        fn.__annotations__ = self.method.__annotations__
        fn.__name__ = self.method.__name__
        setattr(fn, "__signature__", inspect.signature(self.method))
        route = getattr(app, self.name)

        @wraps(signature_proxy_function)
        async def handle_request(*args, **kwargs):
            async def fn(*args, **kwargs):
                request_id = str(uuid4()).split("-")[0]
                request_queue.put(
                    {
                        "__type__": "request",
                        "name": self.flow_name,
                        "method_name": self.method.__name__,
                        "args": args,
                        "kwargs": kwargs,
                        "id": request_id,
                    }
                )

                t0 = time.time()
                while request_id not in commands_response_store:
                    await asyncio.sleep(0.1)
                    if (time.time() - t0) > self.timeout:
                        raise Exception("The response was never received.")

                return commands_response_store.pop(request_id)

            return await asyncio.create_task(fn(*args, **kwargs))

        route(self.route, **self.kwargs)(handle_request)


class Post(Protocol):

    name = "post"


class Get(Protocol):

    name = "get"


class Put(Protocol):

    name = "put"


class Delete(Protocol):

    name = "delete"
