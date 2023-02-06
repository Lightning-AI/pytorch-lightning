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
import json
import os
import queue
import sys
import traceback
from copy import deepcopy
from multiprocessing import Queue
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from time import sleep
from typing import Dict, List, Mapping, Optional, Union

import uvicorn
from deepdiff import DeepDiff, Delta
from fastapi import FastAPI, File, HTTPException, Request, Response, status, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed

from lightning_app.api.http_methods import _HttpMethod
from lightning_app.api.request_types import _DeltaRequest
from lightning_app.core.constants import (
    ENABLE_PULLING_STATE_ENDPOINT,
    ENABLE_PUSHING_STATE_ENDPOINT,
    ENABLE_STATE_WEBSOCKET,
    ENABLE_UPLOAD_ENDPOINT,
    FRONTEND_DIR,
    get_cloud_queue_type,
)
from lightning_app.core.queues import QueuingSystem
from lightning_app.storage import Drive
from lightning_app.utilities.app_helpers import InMemoryStateStore, Logger, StateStore
from lightning_app.utilities.app_status import AppStatus
from lightning_app.utilities.cloud import is_running_in_cloud
from lightning_app.utilities.component import _context
from lightning_app.utilities.enum import ComponentContext, OpenAPITags
from lightning_app.utilities.imports import _is_starsessions_available

if _is_starsessions_available():
    from starsessions import SessionMiddleware
else:

    class SessionMiddleware:
        pass


# TODO: fixed uuid for now, it will come from the FastAPI session
TEST_SESSION_UUID = "1234"

STATE_EVENT = "State changed"

frontend_static_dir = os.path.join(FRONTEND_DIR, "static")

api_app_delta_queue: Queue = None

template = {"ui": {}, "app": {}}
templates = Jinja2Templates(directory=FRONTEND_DIR)

# TODO: try to avoid using global var for state store
global_app_state_store = InMemoryStateStore()
global_app_state_store.add(TEST_SESSION_UUID)

lock = Lock()

app_spec: Optional[List] = None
app_status: Optional[AppStatus] = None
app_annotations: Optional[List] = None

# In the future, this would be abstracted to support horizontal scaling.
responses_store = {}

logger = Logger(__name__)

# This can be replaced with a consumer that publishes states in a kv-store
# in a serverless architecture


class UIRefresher(Thread):
    def __init__(
        self,
        api_publish_state_queue,
        api_response_queue,
        refresh_interval: float = 0.1,
    ) -> None:
        super().__init__(daemon=True)
        self.api_publish_state_queue = api_publish_state_queue
        self.api_response_queue = api_response_queue
        self._exit_event = Event()
        self.refresh_interval = refresh_interval

    def run(self):
        # TODO: Create multiple threads to handle the background logic
        # TODO: Investigate the use of `parallel=True`
        try:
            while not self._exit_event.is_set():
                self.run_once()
                # Note: Sleep to reduce queue calls.
                sleep(self.refresh_interval)
        except Exception as e:
            logger.error(traceback.print_exc())
            raise e

    def run_once(self):
        try:
            global app_status
            state, app_status = self.api_publish_state_queue.get(timeout=0)
            with lock:
                global_app_state_store.set_app_state(TEST_SESSION_UUID, state)
        except queue.Empty:
            pass

        try:
            responses = self.api_response_queue.get(timeout=0)
            with lock:
                # TODO: Abstract the responses store to support horizontal scaling.
                global responses_store
                for response in responses:
                    responses_store[response["id"]] = response["response"]
        except queue.Empty:
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        self._exit_event.set()
        super().join(timeout)


class StateUpdate(BaseModel):
    state: dict = {}


openapi_tags = [
    {
        "name": OpenAPITags.APP_CLIENT_COMMAND,
        "description": "The App Endpoints to be triggered exclusively from the CLI",
    },
    {
        "name": OpenAPITags.APP_COMMAND,
        "description": "The App Endpoints that can be triggered equally from the CLI or from a Http Request",
    },
    {
        "name": OpenAPITags.APP_API,
        "description": "The App Endpoints that can be triggered exclusively from a Http Request",
    },
]

app = FastAPI(openapi_tags=openapi_tags)

fastapi_service = FastAPI()

fastapi_service.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _is_starsessions_available():
    fastapi_service.add_middleware(SessionMiddleware, secret_key="secret", autoload=True)


# General sequence is:
# * an update is generated in the UI
# * the value and the location in the state (or the whole state, easier)
#   is sent to the REST API along with the session UID
# * the previous state is loaded from the cache, the delta is generated
# * the previous state is set as set_state, the delta is provided as
#   delta
# * the app applies the delta and runs the entry_fn, which eventually
#   leads to another state
# * the new state is published through the API
# * the UI is updated with the new value of the state
# Before the above happens, we need to refactor App so that it doesn't
# rely on timeouts, but on sequences of updates (and alignments between
# ranks)
@fastapi_service.get("/api/v1/state", response_class=JSONResponse)
async def get_state(
    response: Response,
    x_lightning_type: Optional[str] = Header(None),
    x_lightning_session_uuid: Optional[str] = Header(None),
    x_lightning_session_id: Optional[str] = Header(None),
) -> Mapping:
    if x_lightning_session_uuid is None:
        raise Exception("Missing X-Lightning-Session-UUID header")
    if x_lightning_session_id is None:
        raise Exception("Missing X-Lightning-Session-ID header")

    if not ENABLE_PULLING_STATE_ENDPOINT:
        response.status_code = status.HTTP_405_METHOD_NOT_ALLOWED
        return {"status": "failure", "reason": "This endpoint is disabled."}

    with lock:
        x_lightning_session_uuid = TEST_SESSION_UUID
        state = global_app_state_store.get_app_state(x_lightning_session_uuid)
        global_app_state_store.set_served_state(x_lightning_session_uuid, state)
        return state


def _get_component_by_name(component_name: str, state):
    child = state
    for child_name in component_name.split(".")[1:]:
        try:
            child = child["flows"][child_name]
        except KeyError:
            child = child["structures"][child_name]

    if isinstance(child["vars"]["_layout"], list):
        assert len(child["vars"]["_layout"]) == 1
        return child["vars"]["_layout"][0]["target"]
    return child["vars"]["_layout"]["target"]


@fastapi_service.get("/api/v1/layout", response_class=JSONResponse)
async def get_layout() -> Mapping:
    with lock:
        x_lightning_session_uuid = TEST_SESSION_UUID
        state = global_app_state_store.get_app_state(x_lightning_session_uuid)
        global_app_state_store.set_served_state(x_lightning_session_uuid, state)
        layout = deepcopy(state["vars"]["_layout"])
        for la in layout:
            if la["content"].startswith("root."):
                la["content"] = _get_component_by_name(la["content"], state)
        return layout


@fastapi_service.get("/api/v1/spec", response_class=JSONResponse)
async def get_spec(
    response: Response,
    x_lightning_session_uuid: Optional[str] = Header(None),
    x_lightning_session_id: Optional[str] = Header(None),
) -> Union[List, Dict]:
    if x_lightning_session_uuid is None:
        raise Exception("Missing X-Lightning-Session-UUID header")
    if x_lightning_session_id is None:
        raise Exception("Missing X-Lightning-Session-ID header")

    if not ENABLE_PULLING_STATE_ENDPOINT:
        response.status_code = status.HTTP_405_METHOD_NOT_ALLOWED
        return {"status": "failure", "reason": "This endpoint is disabled."}

    global app_spec
    return app_spec or []


@fastapi_service.post("/api/v1/delta")
async def post_delta(
    request: Request,
    response: Response,
    x_lightning_type: Optional[str] = Header(None),
    x_lightning_session_uuid: Optional[str] = Header(None),
    x_lightning_session_id: Optional[str] = Header(None),
) -> Optional[Dict]:
    """This endpoint is used to make an update to the app state using delta diff, mainly used by streamlit to
    update the state."""

    if x_lightning_session_uuid is None:
        raise Exception("Missing X-Lightning-Session-UUID header")
    if x_lightning_session_id is None:
        raise Exception("Missing X-Lightning-Session-ID header")

    if not ENABLE_PUSHING_STATE_ENDPOINT:
        response.status_code = status.HTTP_405_METHOD_NOT_ALLOWED
        return {"status": "failure", "reason": "This endpoint is disabled."}

    body: Dict = await request.json()
    api_app_delta_queue.put(_DeltaRequest(delta=Delta(body["delta"])))


@fastapi_service.post("/api/v1/state")
async def post_state(
    request: Request,
    response: Response,
    x_lightning_type: Optional[str] = Header(None),
    x_lightning_session_uuid: Optional[str] = Header(None),
    x_lightning_session_id: Optional[str] = Header(None),
) -> Optional[Dict]:
    if x_lightning_session_uuid is None:
        raise Exception("Missing X-Lightning-Session-UUID header")
    if x_lightning_session_id is None:
        raise Exception("Missing X-Lightning-Session-ID header")
    # This needs to be sent so that it can be set as last state
    # in app (see sequencing above)
    # Actually: we need to make sure last_state is actually
    # the latest state seen by the UI, that is, the last state
    # ui to the UI from the API, not the last state
    # obtained by the app.
    body: Dict = await request.json()
    x_lightning_session_uuid = TEST_SESSION_UUID

    if not ENABLE_PUSHING_STATE_ENDPOINT:
        response.status_code = status.HTTP_405_METHOD_NOT_ALLOWED
        return {"status": "failure", "reason": "This endpoint is disabled."}

    if "stage" in body:
        last_state = global_app_state_store.get_served_state(x_lightning_session_uuid)
        state = deepcopy(last_state)
        state["app_state"]["stage"] = body["stage"]
        deep_diff = DeepDiff(last_state, state, verbose_level=2)
    else:
        state = body["state"]
        last_state = global_app_state_store.get_served_state(x_lightning_session_uuid)
        deep_diff = DeepDiff(last_state, state, verbose_level=2)
    api_app_delta_queue.put(_DeltaRequest(delta=Delta(deep_diff)))


@fastapi_service.put("/api/v1/upload_file/{filename}")
async def upload_file(response: Response, filename: str, uploaded_file: UploadFile = File(...)):
    if not ENABLE_UPLOAD_ENDPOINT:
        response.status_code = status.HTTP_405_METHOD_NOT_ALLOWED
        return {"status": "failure", "reason": "This endpoint is disabled."}

    with TemporaryDirectory() as tmp:
        drive = Drive(
            "lit://uploaded_files",
            component_name="file_server",
            allow_duplicates=True,
            root_folder=tmp,
        )
        tmp_file = os.path.join(tmp, filename)

        with open(tmp_file, "wb") as f:
            done = False
            while not done:
                # Note: The 8192 number doesn't have a strong reason.
                content = await uploaded_file.read(8192)
                f.write(content)
                done = content == b""

        with _context(ComponentContext.WORK):
            drive.put(filename)
    return f"Successfully uploaded '{filename}' to the Drive"


@fastapi_service.get("/api/v1/status", response_model=AppStatus)
async def get_status() -> AppStatus:
    """Get the current status of the app and works."""
    global app_status
    if app_status is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="App status hasn't been reported yet."
        )
    return app_status


@fastapi_service.get("/api/v1/annotations", response_class=JSONResponse)
async def get_annotations() -> Union[List, Dict]:
    """Get the annotations associated with this app."""
    global app_annotations
    return app_annotations or []


@fastapi_service.get("/healthz", status_code=200)
async def healthz(response: Response):
    """Health check endpoint used in the cloud FastAPI servers to check the status periodically."""
    # check the queue status only if running in cloud
    if is_running_in_cloud():
        queue_obj = QueuingSystem(get_cloud_queue_type()).get_queue(queue_name="healthz")
        # this is only being implemented on Redis Queue. For HTTP Queue, it doesn't make sense to have every single
        # app checking the status of the Queue server
        if not queue_obj.is_running:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"status": "failure", "reason": "Redis is not available"}
    x_lightning_session_uuid = TEST_SESSION_UUID
    state = global_app_state_store.get_app_state(x_lightning_session_uuid)
    global_app_state_store.set_served_state(x_lightning_session_uuid, state)
    if not state:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "failure", "reason": f"State is empty {state}"}
    return {"status": "ok"}


# Creates session websocket connection to notify client about any state changes
# The websocket instance needs to be stored based on session id so it is accessible in the api layer
@fastapi_service.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not ENABLE_STATE_WEBSOCKET:
        await websocket.close()
        return
    try:
        counter = global_app_state_store.counter
        while True:
            if global_app_state_store.counter != counter:
                await websocket.send_text(f"{global_app_state_store.counter}")
                counter = global_app_state_store.counter
                logger.debug("Updated websocket.")
            await asyncio.sleep(0.01)
    except ConnectionClosed:
        logger.debug("Websocket connection closed")
    await websocket.close()


async def api_catch_all(request: Request, full_path: str):
    raise HTTPException(status_code=404, detail="Not found")


# Serve frontend from a static directory using FastAPI
fastapi_service.mount("/static", StaticFiles(directory=frontend_static_dir, check_dir=False), name="static")


async def frontend_route(request: Request, full_path: str):
    if "pytest" in sys.modules:
        return ""
    return templates.TemplateResponse("index.html", {"request": request})


def register_global_routes():
    # Catch-all for nonexistent API routes (since we define a catch-all for client-side routing)
    fastapi_service.get("/api{full_path:path}", response_class=JSONResponse)(api_catch_all)
    fastapi_service.get("/{full_path:path}", response_class=HTMLResponse)(frontend_route)


class LightningUvicornServer(uvicorn.Server):
    has_started_queue = None

    def run(self, sockets=None):
        self.config.setup_event_loop()
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(self.serve(sockets=sockets))
        if self.has_started_queue:
            asyncio.ensure_future(self.check_is_started(self.has_started_queue))
        loop.run_forever()

    async def check_is_started(self, queue):
        while not self.started:
            await asyncio.sleep(0.1)
        queue.put("SERVER_HAS_STARTED")


def start_server(
    api_publish_state_queue,
    api_delta_queue,
    api_response_queue,
    has_started_queue: Optional[Queue] = None,
    host="127.0.0.1",
    port=8000,
    root_path: str = "",
    uvicorn_run: bool = True,
    spec: Optional[List] = None,
    apis: Optional[List[_HttpMethod]] = None,
    app_state_store: Optional[StateStore] = None,
):
    global api_app_delta_queue
    global global_app_state_store
    global app_spec
    global app_annotations

    app_spec = spec
    api_app_delta_queue = api_delta_queue

    if app_state_store is not None:
        global_app_state_store = app_state_store

    global_app_state_store.add(TEST_SESSION_UUID)

    # Load annotations
    annotations_path = Path("lightning-annotations.json").resolve()
    if annotations_path.exists():
        with open(annotations_path) as f:
            app_annotations = json.load(f)

    refresher = UIRefresher(api_publish_state_queue, api_response_queue)
    refresher.setDaemon(True)
    refresher.start()

    if uvicorn_run:
        host = host.split("//")[-1] if "//" in host else host
        if host == "0.0.0.0":
            logger.info("Your app has started.")
        else:
            logger.info(f"Your app has started. View it in your browser: http://{host}:{port}/view")
        if has_started_queue:
            LightningUvicornServer.has_started_queue = has_started_queue
            # uvicorn is doing some uglyness by replacing uvicorn.main by click command.
            sys.modules["uvicorn.main"].Server = LightningUvicornServer

        # Register the user API.
        if apis:
            for api in apis:
                api.add_route(fastapi_service, api_app_delta_queue, responses_store)

        register_global_routes()

        uvicorn.run(app=fastapi_service, host=host, port=port, log_level="error", root_path=root_path)

    return refresher
