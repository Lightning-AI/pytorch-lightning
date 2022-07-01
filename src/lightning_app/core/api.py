import asyncio
import logging
import os
import queue
import sys
from copy import deepcopy
from multiprocessing import Queue
from threading import Event, Lock, Thread
from typing import Dict, List, Mapping, Optional

import uvicorn
from deepdiff import DeepDiff, Delta
from fastapi import FastAPI, HTTPException, Request, Response, status, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed

from lightning_app.core.constants import FRONTEND_DIR
from lightning_app.core.queues import RedisQueue
from lightning_app.utilities.app_helpers import InMemoryStateStore, StateStore
from lightning_app.utilities.imports import _is_redis_available, _is_starsessions_available

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

logger = logging.getLogger(__name__)


# This can be replaced with a consumer that publishes states in a kv-store
# in a serverless architecture


class UIRefresher(Thread):
    def __init__(self, api_publish_state_queue) -> None:
        super().__init__(daemon=True)
        self.api_publish_state_queue = api_publish_state_queue
        self._exit_event = Event()

    def run(self):
        # TODO: Create multiple threads to handle the background logic
        # TODO: Investigate the use of `parallel=True`
        while not self._exit_event.is_set():
            self.run_once()

    def run_once(self):
        try:
            state = self.api_publish_state_queue.get(timeout=0)
            with lock:
                global_app_state_store.set_app_state(TEST_SESSION_UUID, state)
        except queue.Empty:
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        self._exit_event.set()
        super().join(timeout)


class StateUpdate(BaseModel):
    state: dict = {}


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
    x_lightning_type: Optional[str] = Header(None),
    x_lightning_session_uuid: Optional[str] = Header(None),
    x_lightning_session_id: Optional[str] = Header(None),
) -> Mapping:
    if x_lightning_session_uuid is None:
        raise Exception("Missing X-Lightning-Session-UUID header")
    if x_lightning_session_id is None:
        raise Exception("Missing X-Lightning-Session-ID header")

    with lock:
        x_lightning_session_uuid = TEST_SESSION_UUID
        state = global_app_state_store.get_app_state(x_lightning_session_uuid)
        global_app_state_store.set_served_state(x_lightning_session_uuid, state)
        return state


@fastapi_service.get("/api/v1/spec", response_class=JSONResponse)
async def get_spec(
    x_lightning_session_uuid: Optional[str] = Header(None),
    x_lightning_session_id: Optional[str] = Header(None),
) -> List:
    if x_lightning_session_uuid is None:
        raise Exception("Missing X-Lightning-Session-UUID header")
    if x_lightning_session_id is None:
        raise Exception("Missing X-Lightning-Session-ID header")
    global app_spec
    return app_spec or []


@fastapi_service.post("/api/v1/delta")
async def post_delta(
    request: Request,
    x_lightning_type: Optional[str] = Header(None),
    x_lightning_session_uuid: Optional[str] = Header(None),
    x_lightning_session_id: Optional[str] = Header(None),
) -> Mapping:
    """This endpoint is used to make an update to the app state using delta diff, mainly used by streamlit to update the
    state."""

    if x_lightning_session_uuid is None:
        raise Exception("Missing X-Lightning-Session-UUID header")
    if x_lightning_session_id is None:
        raise Exception("Missing X-Lightning-Session-ID header")

    body: Dict = await request.json()
    delta = body["delta"]
    update_delta = Delta(delta)
    api_app_delta_queue.put(update_delta)


@fastapi_service.post("/api/v1/state")
async def post_state(
    request: Request,
    x_lightning_type: Optional[str] = Header(None),
    x_lightning_session_uuid: Optional[str] = Header(None),
    x_lightning_session_id: Optional[str] = Header(None),
) -> Mapping:
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

    if "stage" in body:
        last_state = global_app_state_store.get_served_state(x_lightning_session_uuid)
        state = deepcopy(last_state)
        state["app_state"]["stage"] = body["stage"]
        deep_diff = DeepDiff(last_state, state)
    else:
        state = body["state"]
        last_state = global_app_state_store.get_served_state(x_lightning_session_uuid)
        deep_diff = DeepDiff(last_state, state)
    update_delta = Delta(deep_diff)
    api_app_delta_queue.put(update_delta)


@fastapi_service.get("/healthz", status_code=200)
async def healthz(response: Response):
    """Health check endpoint used in the cloud FastAPI servers to check the status periodically. This requires Redis to
    be installed for it to work.

    # TODO - Once the state store abstraction is in, check that too
    """
    if not _is_redis_available():
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "failure", "reason": "Redis is not available"}
    if not RedisQueue(name="ping", default_timeout=1).ping():
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


# Catch-all for nonexistent API routes (since we define a catch-all for client-side routing)
@fastapi_service.get("/api{full_path:path}", response_class=JSONResponse)
async def api_catch_all(request: Request, full_path: str):
    raise HTTPException(status_code=404, detail="Not found")


# Serve frontend from a static directory using FastAPI
fastapi_service.mount("/static", StaticFiles(directory=frontend_static_dir, check_dir=False), name="static")


# Catch-all for frontend routes, must be defined after all other routes
@fastapi_service.get("/{full_path:path}", response_class=HTMLResponse)
async def frontend_route(request: Request, full_path: str):
    if "pytest" in sys.modules:
        return ""
    return templates.TemplateResponse("index.html", {"request": request})


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
    has_started_queue: Optional[Queue] = None,
    host="127.0.0.1",
    port=8000,
    uvicorn_run: bool = True,
    spec: Optional[List] = None,
    app_state_store: Optional[StateStore] = None,
):
    global api_app_delta_queue
    global global_app_state_store
    global app_spec
    app_spec = spec
    api_app_delta_queue = api_delta_queue

    if app_state_store is not None:
        global_app_state_store = app_state_store

    global_app_state_store.add(TEST_SESSION_UUID)

    refresher = UIRefresher(api_publish_state_queue)
    refresher.setDaemon(True)
    refresher.start()

    if uvicorn_run:
        host = host.split("//")[-1] if "//" in host else host
        logger.info(f"Your app has started. View it in your browser: http://{host}:{port}/view")
        if has_started_queue:
            LightningUvicornServer.has_started_queue = has_started_queue
            # uvicorn is doing some uglyness by replacing uvicorn.main by click command.
            sys.modules["uvicorn.main"].Server = LightningUvicornServer
        uvicorn.run(app=fastapi_service, host=host, port=port, log_level="error")

    return refresher
