import asyncio
import logging
import os
import secrets
import time
import uuid
from base64 import b64encode
from itertools import cycle
from typing import Any, Dict, List, Tuple, Type

import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED

from lightning_app.core.flow import LightningFlow
from lightning_app.core.work import LightningWork
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.imports import _is_aiohttp_available, requires
from lightning_app.utilities.packaging.cloud_compute import CloudCompute

if _is_aiohttp_available():
    import aiohttp
    import aiohttp.client_exceptions

logger = Logger(__name__)


def _raise_granular_exception(exception: Exception) -> None:
    """Handle an exception from hitting the model servers."""
    if not isinstance(exception, Exception):
        return

    if isinstance(exception, HTTPException):
        raise exception

    if isinstance(exception, aiohttp.client_exceptions.ServerDisconnectedError):
        raise HTTPException(500, "Worker Server Disconnected") from exception

    if isinstance(exception, aiohttp.client_exceptions.ClientError):
        logging.exception(exception)
        raise HTTPException(500, "Worker Server error") from exception

    if isinstance(exception, asyncio.TimeoutError):
        raise HTTPException(408, "Request timed out") from exception

    if isinstance(exception, Exception):
        if exception.args[0] == "Server disconnected":
            raise HTTPException(500, "Worker Server disconnected") from exception

    logging.exception(exception)
    raise HTTPException(500, exception.args[0]) from exception


class _SysInfo(BaseModel):
    num_workers: int
    servers: List[str]
    num_requests: int
    processing_time: int
    global_request_count: int


class _BatchRequestModel(BaseModel):
    inputs: List[Any]


def _create_fastapi(title: str) -> FastAPI:
    fastapi_app = FastAPI(title=title)

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.global_request_count = 0
    fastapi_app.num_current_requests = 0
    fastapi_app.last_processing_time = 0

    @fastapi_app.get("/", include_in_schema=False)
    async def docs():
        return RedirectResponse("/docs")

    @fastapi_app.get("/num-requests")
    async def num_requests() -> int:
        return fastapi_app.num_current_requests

    return fastapi_app


class _LoadBalancer(LightningWork):
    r"""The LoadBalancer is a LightningWork component that collects the requests and sends them to the prediciton API
    asynchronously using RoundRobin scheduling. It also performs auto batching of the incoming requests.

    The LoadBalancer exposes system endpoints with a basic HTTP authentication, in order to activate the authentication
    you need to provide a system password from environment variable::

        lightning run app app.py --env AUTO_SCALER_AUTH_PASSWORD=PASSWORD

    After enabling you will require to send username and password from the request header for the private endpoints.

    Args:
        input_type: Input type.
        output_type: Output type.
        endpoint: The REST API path.
        max_batch_size: The number of requests processed at once.
        timeout_batching: The number of seconds to wait before sending the requests to process in order to allow for
            requests to be batched. In any case, requests are processed as soon as `max_batch_size` is reached.
        timeout_keep_alive: The number of seconds until it closes Keep-Alive connections if no new data is received.
        timeout_inference_request: The number of seconds to wait for inference.
        \**kwargs: Arguments passed to :func:`LightningWork.init` like ``CloudCompute``, ``BuildConfig``, etc.
    """

    @requires(["aiohttp"])
    def __init__(
        self,
        input_type: BaseModel,
        output_type: BaseModel,
        endpoint: str,
        max_batch_size: int = 8,
        # all timeout args are in seconds
        timeout_batching: int = 1,
        timeout_keep_alive: int = 60,
        timeout_inference_request: int = 60,
        **kwargs: Any,
    ) -> None:
        super().__init__(cloud_compute=CloudCompute("default"), **kwargs)
        self._input_type = input_type
        self._output_type = output_type
        self._timeout_keep_alive = timeout_keep_alive
        self._timeout_inference_request = timeout_inference_request
        self.servers = []
        self.max_batch_size = max_batch_size
        self.timeout_batching = timeout_batching
        self._iter = None
        self._batch = []
        self._responses = {}  # {request_id: response}
        self._last_batch_sent = 0

        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        self.endpoint = endpoint

    async def send_batch(self, batch: List[Tuple[str, _BatchRequestModel]]):
        server = next(self._iter)  # round-robin
        request_data: List[_LoadBalancer._input_type] = [b[1] for b in batch]
        batch_request_data = _BatchRequestModel(inputs=request_data)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "accept": "application/json",
                    "Content-Type": "application/json",
                }
                async with session.post(
                    f"{server}{self.endpoint}",
                    json=batch_request_data.dict(),
                    timeout=self._timeout_inference_request,
                    headers=headers,
                ) as response:
                    if response.status == 408:
                        raise HTTPException(408, "Request timed out")
                    response.raise_for_status()
                    response = await response.json()
                    outputs = response["outputs"]
                    if len(batch) != len(outputs):
                        raise RuntimeError(f"result has {len(outputs)} items but batch is {len(batch)}")
                    result = {request[0]: r for request, r in zip(batch, outputs)}
                    self._responses.update(result)
        except Exception as ex:
            result = {request[0]: ex for request in batch}
            self._responses.update(result)

    async def consumer(self):
        while True:
            await asyncio.sleep(0.05)

            batch = self._batch[: self.max_batch_size]
            while batch and (
                (len(batch) == self.max_batch_size) or ((time.time() - self._last_batch_sent) > self.timeout_batching)
            ):
                asyncio.create_task(self.send_batch(batch))

                self._batch = self._batch[self.max_batch_size :]
                batch = self._batch[: self.max_batch_size]
                self._last_batch_sent = time.time()

    async def process_request(self, data: BaseModel):
        if not self.servers:
            raise HTTPException(500, "None of the workers are healthy!")

        request_id = uuid.uuid4().hex
        request: Tuple = (request_id, data)
        self._batch.append(request)

        while True:
            await asyncio.sleep(0.05)

            if request_id in self._responses:
                result = self._responses[request_id]
                del self._responses[request_id]
                _raise_granular_exception(result)
                return result

    def run(self):
        logger.info(f"servers: {self.servers}")
        lock = asyncio.Lock()

        self._iter = cycle(self.servers)
        self._last_batch_sent = time.time()

        fastapi_app = _create_fastapi("Load Balancer")
        security = HTTPBasic()
        fastapi_app.SEND_TASK = None

        @fastapi_app.middleware("http")
        async def current_request_counter(request: Request, call_next):
            if not request.scope["path"] == self.endpoint:
                return await call_next(request)
            fastapi_app.global_request_count += 1
            fastapi_app.num_current_requests += 1
            start_time = time.time()
            response = await call_next(request)
            processing_time = time.time() - start_time
            fastapi_app.last_processing_time = processing_time
            fastapi_app.num_current_requests -= 1
            return response

        @fastapi_app.on_event("startup")
        async def startup_event():
            fastapi_app.SEND_TASK = asyncio.create_task(self.consumer())

        @fastapi_app.on_event("shutdown")
        def shutdown_event():
            fastapi_app.SEND_TASK.cancel()

        def authenticate_private_endpoint(credentials: HTTPBasicCredentials = Depends(security)):
            AUTO_SCALER_AUTH_PASSWORD = os.environ.get("AUTO_SCALER_AUTH_PASSWORD", "")
            if len(AUTO_SCALER_AUTH_PASSWORD) == 0:
                logger.warn(
                    "You have not set a password for private endpoints! To set a password, add "
                    "`--env AUTO_SCALER_AUTH_PASSWORD=<your pass>` to your lightning run command."
                )
            current_password_bytes = credentials.password.encode("utf8")
            is_correct_password = secrets.compare_digest(
                current_password_bytes, AUTO_SCALER_AUTH_PASSWORD.encode("utf8")
            )
            if not is_correct_password:
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect password",
                    headers={"WWW-Authenticate": "Basic"},
                )
            return True

        @fastapi_app.get("/system/info", response_model=_SysInfo)
        async def sys_info(authenticated: bool = Depends(authenticate_private_endpoint)):
            return _SysInfo(
                num_workers=len(self.servers),
                servers=self.servers,
                num_requests=fastapi_app.num_current_requests,
                processing_time=fastapi_app.last_processing_time,
                global_request_count=fastapi_app.global_request_count,
            )

        @fastapi_app.put("/system/update-servers")
        async def update_servers(servers: List[str], authenticated: bool = Depends(authenticate_private_endpoint)):
            async with lock:
                self.servers = servers
            self._iter = cycle(self.servers)

        @fastapi_app.post(self.endpoint, response_model=self._output_type)
        async def balance_api(inputs: self._input_type):
            return await self.process_request(inputs)

        uvicorn.run(
            fastapi_app,
            host=self.host,
            port=self.port,
            loop="uvloop",
            timeout_keep_alive=self._timeout_keep_alive,
            access_log=False,
        )

    def update_servers(self, server_works: List[LightningWork]):
        """Updates works that load balancer distributes requests to.

        AutoScaler uses this method to increase/decrease the number of works.
        """
        old_servers = set(self.servers)
        server_urls: List[str] = [server.url for server in server_works if server.url]
        new_servers = set(server_urls)

        if new_servers == old_servers:
            return

        if new_servers - old_servers:
            logger.info(f"servers added: {new_servers - old_servers}")

        deleted_servers = old_servers - new_servers
        if deleted_servers:
            logger.info(f"servers deleted: {deleted_servers}")

        self.send_request_to_update_servers(server_urls)

    def send_request_to_update_servers(self, servers: List[str]):
        AUTHORIZATION_TYPE = "Basic"
        USERNAME = "lightning"
        AUTO_SCALER_AUTH_PASSWORD = os.environ.get("AUTO_SCALER_AUTH_PASSWORD", "")

        try:
            param = f"{USERNAME}:{AUTO_SCALER_AUTH_PASSWORD}".encode()
            data = b64encode(param).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as e:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Basic"},
            ) from e
        headers = {
            "accept": "application/json",
            "username": USERNAME,
            "Authorization": AUTHORIZATION_TYPE + " " + data,
        }
        response = requests.put(f"{self.url}/system/update-servers", json=servers, headers=headers, timeout=10)
        response.raise_for_status()


class AutoScaler(LightningFlow):
    """The ``AutoScaler`` can be used to automatically change the number of replicas of the given server in
    response to changes in the number of incoming requests. Incoming requests will be batched and balanced across
    the replicas.

    Args:
        min_replicas: The number of works to start when app initializes.
        max_replicas: The max number of works to spawn to handle the incoming requests.
        autoscale_interval: The number of seconds to wait before checking whether to upscale or downscale the works.
        endpoint: Provide the REST API path.
        max_batch_size: (auto-batching) The number of requests to process at once.
        timeout_batching: (auto-batching) The number of seconds to wait before sending the requests to process.
        input_type: Input type.
        output_type: Output type.

    .. testcode::

        import lightning as L

        # Example 1: Auto-scaling serve component out-of-the-box
        app = L.LightningApp(
            L.app.components.AutoScaler(
                MyPythonServer,
                min_replicas=1,
                max_replicas=8,
                autoscale_interval=10,
            )
        )

        # Example 2: Customizing the scaling logic
        class MyAutoScaler(L.app.components.AutoScaler):
            def scale(self, replicas: int, metrics: dict) -> int:
                pending_requests_per_running_or_pending_work = metrics["pending_requests"] / (
                    replicas + metrics["pending_works"]
                )

                # upscale
                max_requests_per_work = self.max_batch_size
                if pending_requests_per_running_or_pending_work >= max_requests_per_work:
                    return replicas + 1

                # downscale
                min_requests_per_work = max_requests_per_work * 0.25
                if pending_requests_per_running_or_pending_work < min_requests_per_work:
                    return replicas - 1

                return replicas


        app = L.LightningApp(
            MyAutoScaler(
                MyPythonServer,
                min_replicas=1,
                max_replicas=8,
                autoscale_interval=10,
                max_batch_size=8,  # for auto batching
                timeout_batching=1,  # for auto batching
            )
        )
    """

    def __init__(
        self,
        work_cls: Type[LightningWork],
        min_replicas: int = 1,
        max_replicas: int = 4,
        autoscale_interval: int = 10,
        max_batch_size: int = 8,
        timeout_batching: float = 1,
        endpoint: str = "api/predict",
        input_type: BaseModel = Dict,
        output_type: BaseModel = Dict,
        *work_args: Any,
        **work_kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_replicas = 0
        self._work_registry = {}

        self._work_cls = work_cls
        self._work_args = work_args
        self._work_kwargs = work_kwargs

        self._input_type = input_type
        self._output_type = output_type
        self.autoscale_interval = autoscale_interval
        self.max_batch_size = max_batch_size

        if max_replicas < min_replicas:
            raise ValueError(
                f"`max_replicas={max_replicas}` must be less than or equal to `min_replicas={min_replicas}`."
            )
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self._last_autoscale = time.time()
        self.fake_trigger = 0

        self.load_balancer = _LoadBalancer(
            input_type=self._input_type,
            output_type=self._output_type,
            endpoint=endpoint,
            max_batch_size=max_batch_size,
            timeout_batching=timeout_batching,
            cache_calls=True,
            parallel=True,
        )
        for _ in range(min_replicas):
            work = self.create_work()
            self.add_work(work)

    @property
    def workers(self) -> List[LightningWork]:
        return [self.get_work(i) for i in range(self.num_replicas)]

    def create_work(self) -> LightningWork:
        """Replicates a LightningWork instance with args and kwargs provided via ``__init__``."""
        cloud_compute = self._work_kwargs.get("cloud_compute", None)
        self._work_kwargs.update(
            dict(
                # TODO: Remove `start_with_flow=False` for faster initialization on the cloud
                start_with_flow=False,
                # don't try to create multiple works in a single machine
                cloud_compute=cloud_compute.clone() if cloud_compute else None,
            )
        )
        return self._work_cls(*self._work_args, **self._work_kwargs)

    def add_work(self, work) -> str:
        """Adds a new LightningWork instance.

        Returns:
            The name of the new work attribute.
        """
        work_attribute = uuid.uuid4().hex
        work_attribute = f"worker_{self.num_replicas}_{str(work_attribute)}"
        setattr(self, work_attribute, work)
        self._work_registry[self.num_replicas] = work_attribute
        self.num_replicas += 1
        return work_attribute

    def remove_work(self, index: int) -> str:
        """Removes the ``index`` th LightningWork instance."""
        work_attribute = self._work_registry[index]
        del self._work_registry[index]
        work = getattr(self, work_attribute)
        work.stop()
        self.num_replicas -= 1
        return work_attribute

    def get_work(self, index: int) -> LightningWork:
        """Returns the ``LightningWork`` instance with the given index."""
        work_attribute = self._work_registry[index]
        work = getattr(self, work_attribute)
        return work

    def run(self):
        if not self.load_balancer.is_running:
            self.load_balancer.run()

        for work in self.workers:
            work.run()

        if self.load_balancer.url:
            self.fake_trigger += 1  # Note: change state to keep calling `run`.
            self.autoscale()

    def scale(self, replicas: int, metrics: dict) -> int:
        """The default scaling logic that users can override.

        Args:
            replicas: The number of running works.
            metrics: ``metrics['pending_requests']`` is the total number of requests that are currently pending.
                ``metrics['pending_works']`` is the number of pending works.

        Returns:
            The target number of running works. The value will be adjusted after this method runs
            so that it satisfies ``min_replicas<=replicas<=max_replicas``.
        """
        pending_requests_per_running_or_pending_work = metrics["pending_requests"] / (
            replicas + metrics["pending_works"]
        )

        # scale out if the number of pending requests exceeds max batch size.
        max_requests_per_work = self.max_batch_size
        if pending_requests_per_running_or_pending_work >= max_requests_per_work:
            return replicas + 1

        # scale in if the number of pending requests is below 25% of max_requests_per_work
        min_requests_per_work = max_requests_per_work * 0.25
        if pending_requests_per_running_or_pending_work < min_requests_per_work:
            return replicas - 1

        return replicas

    @property
    def num_pending_requests(self) -> int:
        """Fetches the number of pending requests via load balancer."""
        return int(requests.get(f"{self.load_balancer.url}/num-requests").json())

    @property
    def num_pending_works(self) -> int:
        """The number of pending works."""
        return sum(work.is_pending for work in self.workers)

    def autoscale(self) -> None:
        """Adjust the number of works based on the target number returned by ``self.scale``."""
        if time.time() - self._last_autoscale < self.autoscale_interval:
            return

        self.load_balancer.update_servers(self.workers)

        metrics = {
            "pending_requests": self.num_pending_requests,
            "pending_works": self.num_pending_works,
        }

        # ensure min_replicas <= num_replicas <= max_replicas
        num_target_workers = max(
            self.min_replicas,
            min(self.max_replicas, self.scale(self.num_replicas, metrics)),
        )

        # upscale
        num_workers_to_add = num_target_workers - self.num_replicas
        for _ in range(num_workers_to_add):
            logger.info(f"Upscaling from {self.num_replicas} to {self.num_replicas + 1}")
            work = self.create_work()
            new_work_id = self.add_work(work)
            logger.info(f"Work created: '{new_work_id}'")

        # downscale
        num_workers_to_remove = self.num_replicas - num_target_workers
        for _ in range(num_workers_to_remove):
            logger.info(f"Downscaling from {self.num_replicas} to {self.num_replicas - 1}")
            removed_work_id = self.remove_work(self.num_replicas - 1)
            logger.info(f"Work removed: '{removed_work_id}'")

        self.load_balancer.update_servers(self.workers)
        self._last_autoscale = time.time()

    def configure_layout(self):
        tabs = [{"name": "Swagger", "content": self.load_balancer.url}]
        return tabs
