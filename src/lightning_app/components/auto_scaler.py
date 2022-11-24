import asyncio
import logging
import os
import time
import uuid
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import aiohttp.client_exceptions
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from lightning_app.core.flow import LightningFlow
from lightning_app.core.work import LightningWork
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.packaging.cloud_compute import CloudCompute

MIN_REPLICA = int(os.environ.get("MUSE_MIN_WORKERS", 1))
KEEP_ALIVE_TIMEOUT = float(os.environ.get("KEEP_ALIVE_TIMEOUT", 60))
INFERENCE_REQUEST_TIMEOUT = float(os.environ.get("KEEP_ALIVE_TIMEOUT", 60))
OPEN_PROMPTS = None

logger = Logger(__name__)


def raise_granular_exception(exception: Exception):
    """handle the exceptions coming from hitting the model servers."""
    if not isinstance(exception, Exception):
        return

    if isinstance(exception, HTTPException):
        raise exception

    if isinstance(exception, aiohttp.client_exceptions.ServerDisconnectedError):
        raise HTTPException(500, "Worker Server Disconnected")

    if isinstance(exception, aiohttp.client_exceptions.ClientError):
        logging.exception(exception)
        raise HTTPException(500, "Worker Server error")

    if isinstance(exception, asyncio.TimeoutError):
        raise TimeoutException()

    if isinstance(exception, Exception):
        if exception.args[0] == "Server disconnected":
            raise HTTPException(500, "Worker Server disconnected")

    logging.exception(exception)
    raise HTTPException(500, exception.args[0])


class TimeoutException(HTTPException):
    def __init__(self, status_code=408, detail="Request timed out.", *args, **kwargs):
        super().__init__(status_code=status_code, detail=detail, *args, **kwargs)


class LimitBacklogException(HTTPException):
    def __init__(self, status_code=408, detail="Model Server has too much backlog.", *args, **kwargs):
        super().__init__(status_code=status_code, detail=detail, *args, **kwargs)


class SysInfo(BaseModel):
    num_workers: int
    servers: List[str]
    num_requests: int
    process_time: int
    global_request_count: int


class BatchRequestModel(BaseModel):
    inputs: List[Any]


class BatchResponse(BaseModel):
    outputs: List[Any]


def create_fastapi(title: str) -> FastAPI:
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
    fastapi_app.last_process_time = 0

    @fastapi_app.middleware("http")
    async def current_request_counter(request: Request, call_next):
        if not request.scope["path"] == "/api/predict":
            return await call_next(request)
        fastapi_app.global_request_count += 1
        fastapi_app.num_current_requests += 1
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        fastapi_app.last_process_time = process_time
        fastapi_app.num_current_requests -= 1
        return response

    @fastapi_app.get("/", include_in_schema=False)
    async def docs():
        return RedirectResponse("/docs")

    @fastapi_app.get("/num-requests")
    async def num_requests() -> int:
        return fastapi_app.num_current_requests

    return fastapi_app


class LoadBalancer(LightningWork):
    r"""The LoadBalancer is a LightningWork component that collects the requests and sends it to the prediciton API
    asynchronously using RoundRobin scheduling. It also performs auto batching of the incoming requests.

    The LoadBalancer exposes system endpoints with a basic HTTP authentication, in order to activate the authentication
    you need to provide a system password from environment variable
    `lightning run app lb_flow.py --env MUSE_SYSTEM_PASSWORD=PASSWORD`.
    After enabling you will require to send username and password from the request header for the private endpoints.

    Args:
        max_batch_size: Number of requests processed at once.
        batch_timeout_secs: Number of seconds to wait before sending the requests to process.
        \**kwargs: Arguments passed to :func:`LightningWork.init` like ``CloudCompute``, ``BuildConfig``, etc.
    """

    def __init__(self, input_schema, output_schema, worker_url: str, max_batch_size=8, batch_timeout_secs=10, **kwargs):
        super().__init__(cloud_compute=CloudCompute("default"), **kwargs)
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._server_ready = False
        self.servers = []
        self.max_batch_size = max_batch_size
        self.batch_timeout_secs = batch_timeout_secs
        self._ITER = None
        self._batch = []
        self._responses = {}  # {request_id: response}
        self._last_batch_sent = 0
        self.worker_url = worker_url

    async def send_batch(self, batch: List[Tuple[str, BatchRequestModel]]):
        # unit method
        server = next(self._ITER)
        request_data: List[LoadBalancer._input_schema] = [b[1] for b in batch]
        batch_request_data = BatchRequestModel(inputs=request_data)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "accept": "application/json",
                    "Content-Type": "application/json",
                }
                async with session.post(
                    f"{server}/{self.worker_url}",
                    json=batch_request_data.dict(),
                    timeout=INFERENCE_REQUEST_TIMEOUT,
                    headers=headers,
                ) as response:
                    if response.status == 408:
                        raise TimeoutException()
                    response.raise_for_status()
                    response = await response.json()
                    outputs = response["outputs"]
                    assert len(batch) == len(outputs), f"result has {len(outputs)} items but batch is {len(batch)}"
                    result = {request[0]: r for request, r in zip(batch, outputs)}
                    self._responses.update(result)
        except Exception as e:
            result = {request[0]: e for request in batch}
            self._responses.update(result)

    async def consumer(self):
        while True:
            await asyncio.sleep(0.05)

            has_sent = False

            batch = self._batch[: self.max_batch_size]
            while batch and (
                (len(batch) >= self.max_batch_size) or ((time.time() - self._last_batch_sent) > self.batch_timeout_secs)
            ):
                has_sent = True

                asyncio.create_task(self.send_batch(batch))

                self._batch = self._batch[self.max_batch_size :]
                batch = self._batch[: self.max_batch_size]

            if has_sent:
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
                raise_granular_exception(result)
                return result

    def run(self):
        if self._server_ready:
            return

        INPUT_SCHEMA = self._input_schema
        OUTPUT_SCHEMA = self._output_schema

        logger.info(f"servers: {self.servers}")

        self._ITER = cycle(self.servers)
        self._last_batch_sent = time.time()

        fastapi_app = create_fastapi("Load Balancer")
        fastapi_app.global_request_count = 0
        fastapi_app.num_current_requests = 0
        fastapi_app.last_process_time = 0
        fastapi_app.SEND_TASK = None

        @fastapi_app.on_event("startup")
        async def startup_event():
            fastapi_app.SEND_TASK = asyncio.create_task(self.consumer())
            self._server_ready = True

        @fastapi_app.on_event("shutdown")
        def shutdown_event():
            fastapi_app.SEND_TASK.cancel()
            self._server_ready = False

        @fastapi_app.get("/system/info", response_model=SysInfo)
        async def sys_info():
            return SysInfo(
                num_workers=len(self.servers),
                servers=self.servers,
                num_requests=fastapi_app.num_current_requests,
                process_time=fastapi_app.last_process_time,
                global_request_count=fastapi_app.global_request_count,
            )

        @fastapi_app.put("/system/update-servers")
        async def update_servers(servers: List[str]):
            self.servers = servers
            self._ITER = cycle(self.servers)

        @fastapi_app.post("/api/predict", response_model=OUTPUT_SCHEMA)
        async def balance_api(inputs: INPUT_SCHEMA):
            return await self.process_request(inputs)

        uvicorn.run(
            fastapi_app,
            host=self.host,
            port=self.port,
            loop="uvloop",
            timeout_keep_alive=KEEP_ALIVE_TIMEOUT,
            access_log=False,
        )

    def update_servers(self, server_works: List[LightningWork]):
        old_servers = set(self.servers)
        server_urls: List[str] = [server.url for server in server_works if server.url]
        new_servers = set(server_urls)
        if new_servers == old_servers:
            logging.debug("no new server added")
            return
        if new_servers - old_servers:
            logger.info(f"servers added: {new_servers - old_servers}")

        deleted_servers = old_servers - new_servers
        if deleted_servers:
            logger.info(f"servers added: {deleted_servers}")

        headers = {
            "accept": "application/json",
            "username": "lightning",
        }
        response = requests.put(f"{self.url}/system/update-servers", json=server_urls, headers=headers, timeout=10)
        response.raise_for_status()


class AutoScaler(LightningFlow):
    """A LightningFlow component that handles all the servers and uses load balancer to spawn up and shutdown based
    on current requests in the queue.

    Args:
        min_replica: Number of works to start when app initializes.
        max_replica: Max numbers of works to spawn to handle the incoming requests.
        autoscale_interval: Number of seconds to wait before checking whether to upscale or downscale the works.
        max_batch_size: Number of requests to process at once.
        batch_timeout_secs: Number of seconds to wait before sending the requests to process.
        device_type: GPU type to use for the works.
        downscale_threshold: Lower limit to determine when to stop works.
        upscale_threshold: Upper limit to determine when to spawn up a new work.
        worker_url: Default=api/predict. Provide the REST API path
        input_schema:
        output_schema
    """

    def __init__(
        self,
        work_cls: type,
        min_replica: int = MIN_REPLICA,
        max_replica: int = 4,
        autoscale_interval: int = 1 * 10,
        max_batch_size: int = 8,
        batch_timeout_secs: float = 2,
        downscale_threshold: Optional[int] = None,
        upscale_threshold: Optional[int] = None,
        worker_url: str = None,
        input_schema: Any = Dict,
        output_schema: Any = Dict,
    ):
        super().__init__()
        self._worker_count = 0
        self._work_registry = {}

        self._work_cls = work_cls
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._initial_num_workers = min_replica
        self.autoscale_interval = autoscale_interval
        self.max_workers = max_replica
        self.downscale_threshold = downscale_threshold or min_replica
        self.upscale_threshold = upscale_threshold or min_replica * max_batch_size
        self.fake_trigger = 0
        self._last_autoscale = time.time()

        worker_url = worker_url or "api/predict"
        self.load_balancer = LoadBalancer(
            input_schema=self._input_schema,
            output_schema=self._output_schema,
            worker_url=worker_url,
            max_batch_size=max_batch_size,
            batch_timeout_secs=batch_timeout_secs,
            cache_calls=True,
            parallel=True,
        )
        for _ in range(min_replica):
            work = self.create_worker()
            self.add_work(work)

        logger.info(
            f"LB initialized with min replica={min_replica}, "
            f"max_replica={max_replica}, "
            f"batch timeout={batch_timeout_secs}, "
            f"batch size={max_batch_size}"
        )

    @property
    def workers(self) -> List[LightningWork]:
        works = []
        for i in range(self._worker_count):
            work = self.get_work(i)
            works.append(work)
        return works

    def create_worker(self, *args, **kwargs) -> LightningWork:
        """implement."""
        return self._work_cls()

    def add_work(self, work) -> str:
        work_attribute = uuid.uuid4().hex
        work_attribute = f"worker_{self._worker_count}_{str(work_attribute)}"
        setattr(self, work_attribute, work)
        self._work_registry[self._worker_count] = work_attribute
        self._worker_count += 1
        return work_attribute

    def remove_work(self, index: int) -> str:
        work_attribute = self._work_registry[index]
        del self._work_registry[index]
        work = getattr(self, work_attribute)
        work.stop()
        self._worker_count -= 1
        return work_attribute

    def get_work(self, index: int):
        work_attribute = self._work_registry[index]
        work = getattr(self, work_attribute)
        return work

    def run(self):
        if not self.load_balancer.is_running:
            self.load_balancer.run()

        for worker in self.workers:
            worker.run()

        if self.load_balancer.url:
            self.fake_trigger += 1
            self.autoscale()

    def autoscale(self):
        """Upscale and down scale model inference works based on the number of requests."""
        if time.time() - self._last_autoscale < self.autoscale_interval:
            return

        self.load_balancer.update_servers(self.workers)

        num_requests = int(requests.get(f"{self.load_balancer.url}/num-requests").json())
        num_workers = len(self.workers)

        # upscale
        if num_requests > self.upscale_threshold and num_workers < self.max_workers:
            idx = self._worker_count
            logger.info(f"Upscale to {self._worker_count + 1}")
            work = self.create_worker()
            new_work_id = self.add_work(work)
            logger.info("new work id:", new_work_id)

        # downscale
        elif num_requests < self.downscale_threshold and num_workers > self._initial_num_workers:
            idx = self._worker_count - 1
            logger.info(f"Downscale to {idx}")
            logger.info("prev num servers:", len(self.workers))
            removed_id = self.remove_work(idx)
            logger.info("removed:", removed_id)
            logger.info("new num servers:", len(self.workers))

        self.load_balancer.update_servers(self.workers)
        self._last_autoscale = time.time()

    def configure_layout(self):
        tabs = [{"name": "Swagger", "content": self.load_balancer.url}]
        return tabs
