import asyncio
import contextlib
import json
import logging
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from multiprocessing import Process
from pathlib import Path
from time import sleep, time
from unittest import mock

import aiohttp
import lightning.app
import pytest
import requests
from deepdiff import DeepDiff, Delta
from fastapi import HTTPException, Request
from httpx import AsyncClient
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.api.http_methods import Post
from lightning.app.core import api
from lightning.app.core.api import (
    UIRefresher,
    fastapi_service,
    global_app_state_store,
    register_global_routes,
    start_server,
)
from lightning.app.core.constants import APP_SERVER_PORT
from lightning.app.runners import MultiProcessRuntime
from lightning.app.storage.drive import Drive
from lightning.app.testing.helpers import _MockQueue
from lightning.app.utilities.app_status import AppStatus
from lightning.app.utilities.component import _set_frontend_context, _set_work_context
from lightning.app.utilities.enum import AppStage
from lightning.app.utilities.load_app import extract_metadata_from_app
from lightning.app.utilities.redis import check_if_redis_running
from lightning.app.utilities.state import AppState, headers_for
from pydantic import BaseModel

register_global_routes()


class WorkA(LightningWork):
    def __init__(self):
        super().__init__(parallel=True, start_with_flow=False)
        self.var_a = 0
        self.drive = Drive("lit://test_app_state_api")

    def run(self):
        state = AppState()
        assert state._my_affiliation == ("work_a",)
        # this would download and push data to the REST API.
        assert state.var_a == 0
        assert isinstance(state.drive, Drive)
        assert state.drive.component_name == "root.work_a"

        with open("test_app_state_api.txt", "w") as f:
            f.write("here")
        state.drive.put("test_app_state_api.txt")
        state.var_a = -1


class _A(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_a = WorkA()

    def run(self):
        if self.work_a.var_a == -1:
            self.stop()
        self.work_a.run()


@pytest.mark.skipif(sys.platform == "win32" or sys.platform == "darwin", reason="too slow on Windows or macOs")
def test_app_state_api():
    """This test validates the AppState can properly broadcast changes from work within its own process."""
    app = LightningApp(_A(), log_level="debug")
    MultiProcessRuntime(app, start_server=True).dispatch()
    assert app.root.work_a.var_a == -1
    _set_work_context()
    assert app.root.work_a.drive.list(".") == ["test_app_state_api.txt"]
    _set_frontend_context()
    assert app.root.work_a.drive.list(".") == ["test_app_state_api.txt"]
    os.remove("test_app_state_api.txt")


class A2(LightningFlow):
    def __init__(self):
        super().__init__()
        self.var_a = 0
        self.a = _A()

    def update_state(self):
        state = AppState()
        # this would download and push data to the REST API.
        assert state.a.work_a.var_a == 0
        assert state.var_a == 0
        state.var_a = -1

    def run(self):
        if self.var_a == 0:
            self.update_state()
        elif self.var_a == -1:
            self.stop()


@pytest.mark.skipif(sys.platform == "win32" or sys.platform == "darwin", reason="too slow on Windows or macOs")
def test_app_state_api_with_flows():
    """This test validates the AppState can properly broadcast changes from flows."""
    app = LightningApp(A2(), log_level="debug")
    MultiProcessRuntime(app, start_server=True).dispatch()
    assert app.root.var_a == -1


class NestedFlow(LightningFlow):
    def run(self):
        pass

    def configure_layout(self):
        return {"name": "main", "content": "https://te"}


class FlowA(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.flow = NestedFlow()
        self.dict = lightning.app.structures.Dict(**{"0": NestedFlow()})
        self.list = lightning.app.structures.List(*[NestedFlow()])

    def run(self):
        self.counter += 1
        if self.counter >= 3:
            self.stop()

    def configure_layout(self):
        return [
            {"name": "main_1", "content": "https://te"},
            {"name": "main_2", "content": self.flow},
            {"name": "main_3", "content": self.dict["0"]},
            {"name": "main_4", "content": self.list[0]},
        ]


class AppStageTestingApp(LightningApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter_running = 0
        self.counter_stopped = 0
        self.counter = 0

    def _change_stage(self, enum):
        previous_state = deepcopy(self.state)
        current_state = self.state
        current_state["app_state"]["stage"] = enum.value
        deep_diff = DeepDiff(previous_state, current_state, verbose_level=2)
        self.api_delta_queue.put(Delta(deep_diff))

    def maybe_apply_changes(self):
        if self.counter_stopped == 1 and self.counter_running == 1:
            if self.counter == 0:
                self._change_stage(AppStage.RUNNING)
            self.counter += 1
            if self.counter == 3:
                self._change_stage(AppStage.STOPPING)

        # emulate pending from the UI.
        elif self.stage == AppStage.BLOCKING:
            self._change_stage(AppStage.RUNNING)
            self.counter_running += 1

        elif self.root.counter == 2:
            self._change_stage(AppStage.RESTARTING)
            self.counter_stopped += 1

        super().maybe_apply_changes()


# FIXME: This test doesn't assert anything
@pytest.mark.xfail(strict=False, reason="TODO: Resolve flaky test.")
def test_app_stage_from_frontend():
    """This test validates that delta from the `api_delta_queue` manipulating the ['app_state']['stage'] would start
    and stop the app."""
    app = AppStageTestingApp(FlowA(), log_level="debug")
    app.stage = AppStage.BLOCKING
    MultiProcessRuntime(app, start_server=True).dispatch()


def test_update_publish_state_and_maybe_refresh_ui():
    """This test checks that the method properly:

    - receives the state from the `publish_state_queue` and populates the app_state_store
    - receives a notification to refresh the UI and makes a GET Request (streamlit).

    """
    app = AppStageTestingApp(FlowA(), log_level="debug")
    publish_state_queue = _MockQueue("publish_state_queue")
    api_response_queue = _MockQueue("api_response_queue")

    publish_state_queue.put((app.state_with_changes, None))

    thread = UIRefresher(publish_state_queue, api_response_queue)
    thread.run_once()

    assert global_app_state_store.get_app_state("1234") == app.state_with_changes
    global_app_state_store.remove("1234")
    global_app_state_store.add("1234")


@pytest.mark.parametrize("x_lightning_type", ["DEFAULT", "STREAMLIT"])
@pytest.mark.anyio()
async def test_start_server(x_lightning_type, monkeypatch):
    """This test relies on FastAPI TestClient and validates that the REST API properly provides:

    - the state on GET /api/v1/state
    - push a delta when making a POST request to /api/v1/state

    """

    class InfiniteQueue(_MockQueue):
        def get(self, timeout: int = 0):
            return self._queue[0]

    app = AppStageTestingApp(FlowA(), log_level="debug")
    app._update_layout()
    app.stage = AppStage.BLOCKING
    publish_state_queue = InfiniteQueue("publish_state_queue")
    change_state_queue = _MockQueue("change_state_queue")
    has_started_queue = _MockQueue("has_started_queue")
    api_response_queue = _MockQueue("api_response_queue")
    state = app.state_with_changes
    publish_state_queue.put((state, AppStatus(is_ui_ready=True, work_statuses={})))
    spec = extract_metadata_from_app(app)
    ui_refresher = start_server(
        publish_state_queue,
        change_state_queue,
        api_response_queue,
        has_started_queue=has_started_queue,
        uvicorn_run=False,
        spec=spec,
    )
    headers = headers_for({"type": x_lightning_type})

    async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
        with pytest.raises(Exception, match="X-Lightning-Session-UUID"):
            await client.get("/api/v1/spec")

        with pytest.raises(Exception, match="X-Lightning-Session-ID"):
            await client.get("/api/v1/spec", headers={"X-Lightning-Session-UUID": headers["X-Lightning-Session-UUID"]})

        response = await client.get("/api/v1/spec", headers=headers)
        assert response.json() == spec

        with pytest.raises(Exception, match="X-Lightning-Session-UUID"):
            await client.get("/api/v1/state")

        with pytest.raises(Exception, match="X-Lightning-Session-ID"):
            await client.get("/api/v1/state", headers={"X-Lightning-Session-UUID": headers["X-Lightning-Session-UUID"]})

        response = await client.get("/api/v1/state", headers=headers)
        assert response.json() == state
        assert response.status_code == 200

        new_state = deepcopy(state)
        new_state["vars"]["counter"] += 1

        with pytest.raises(Exception, match="X-Lightning-Session-UUID"):
            await client.post("/api/v1/state")

        with pytest.raises(Exception, match="X-Lightning-Session-ID"):
            await client.post(
                "/api/v1/state", headers={"X-Lightning-Session-UUID": headers["X-Lightning-Session-UUID"]}
            )

        response = await client.post("/api/v1/state", json={"stage": "running"}, headers=headers)
        assert change_state_queue._queue[0].to_dict() == {
            "values_changed": {"root['app_state']['stage']": {"new_value": "running"}}
        }
        assert response.status_code == 200

        response = await client.get("/api/v1/layout")
        assert json.loads(response.json()) == [
            {"name": "main_1", "content": "https://te", "target": "https://te"},
            {"name": "main_2", "content": "https://te"},
            {"name": "main_3", "content": "https://te"},
            {"name": "main_4", "content": "https://te"},
        ]

        response = await client.get("/api/v1/status")
        assert response.json() == {"is_ui_ready": True, "work_statuses": {}}

        response = await client.post("/api/v1/state", json={"state": new_state}, headers=headers)
        assert change_state_queue._queue[1].to_dict() == {
            "values_changed": {"root['vars']['counter']": {"new_value": 1}}
        }
        assert response.status_code == 200

        response = await client.post(
            "/api/v1/delta",
            json={
                "delta": {
                    "values_changed": {"root['flows']['video_search']['vars']['should_process']": {"new_value": True}}
                }
            },
            headers=headers,
        )
        assert change_state_queue._queue[2].to_dict() == {
            "values_changed": {"root['flows']['video_search']['vars']['should_process']": {"new_value": True}}
        }
        assert response.status_code == 200

        monkeypatch.setattr(api, "ENABLE_PULLING_STATE_ENDPOINT", False)

        response = await client.get("/api/v1/state", headers=headers)
        assert response.status_code == 405

        response = await client.post("/api/v1/state", json={"state": new_state}, headers=headers)
        assert response.status_code == 200

        monkeypatch.setattr(api, "ENABLE_PUSHING_STATE_ENDPOINT", False)

        response = await client.post("/api/v1/state", json={"state": new_state}, headers=headers)
        assert response.status_code == 405

        response = await client.post(
            "/api/v1/delta",
            json={
                "delta": {
                    "values_changed": {"root['flows']['video_search']['vars']['should_process']": {"new_value": True}}
                }
            },
            headers=headers,
        )
        assert change_state_queue._queue[2].to_dict() == {
            "values_changed": {"root['flows']['video_search']['vars']['should_process']": {"new_value": True}}
        }
        assert response.status_code == 405

        # used to clean the app_state_store to following test.
        global_app_state_store.remove("1234")
        global_app_state_store.add("1234")

    del client
    ui_refresher.join(0)


@pytest.mark.parametrize(
    ("path", "expected_status_code"), [("/api/v1", 404), ("/api/v1/asdf", 404), ("/api/asdf", 404), ("/api", 404)]
)
@pytest.mark.anyio()
async def test_state_api_routes(path, expected_status_code):
    async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == expected_status_code


@pytest.mark.skipif(not check_if_redis_running(), reason="redis not running")
@pytest.mark.anyio()
async def test_health_endpoint_success():
    global_app_state_store.store = {}
    global_app_state_store.add("1234")
    async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
        # will respond 503 if redis is not running
        response = await client.get("/healthz")
        assert response.status_code == 500
        assert response.json() == {"status": "failure", "reason": "State is empty {}"}
        global_app_state_store.set_app_state("1234", {"state": None})
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        global_app_state_store.remove("1234")
        global_app_state_store.store = {}
        global_app_state_store.add("1234")


@pytest.mark.skipif(
    check_if_redis_running(), reason="this is testing the failure condition " "for which the redis should not run"
)
@pytest.mark.anyio()
async def test_health_endpoint_failure(monkeypatch):
    monkeypatch.setenv("LIGHTNING_APP_STATE_URL", "http://someurl")  # adding this to make is_running_in_cloud pass
    monkeypatch.setitem(os.environ, "LIGHTNING_CLOUD_QUEUE_TYPE", "redis")
    async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
        # will respond 503 if redis is not running
        response = await client.get("/healthz")
        assert response.status_code == 500


@pytest.mark.parametrize(
    ("path", "expected_status_code"),
    [
        ("/", 200),
        ("/asdf", 200),
        ("/view/component_a", 200),
    ],
)
@pytest.mark.anyio()
async def test_frontend_routes(path, expected_status_code):
    async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == expected_status_code


@pytest.mark.xfail(sys.platform == "linux", strict=False, reason="No idea why... need to be fixed")  # fixme
def test_start_server_started():
    """This test ensures has_started_queue receives a signal when the REST API has started."""
    api_publish_state_queue = mp.Queue()
    api_delta_queue = mp.Queue()
    has_started_queue = mp.Queue()
    api_response_queue = mp.Queue()
    kwargs = {
        "api_publish_state_queue": api_publish_state_queue,
        "api_delta_queue": api_delta_queue,
        "has_started_queue": has_started_queue,
        "api_response_queue": api_response_queue,
        "port": 1111,
        "root_path": "",
    }

    server_proc = mp.Process(target=start_server, kwargs=kwargs)
    server_proc.start()
    # requires to wait for the UI to be clicked on.

    # wait for server to be ready
    assert has_started_queue.get() == "SERVER_HAS_STARTED"
    server_proc.kill()


@mock.patch("uvicorn.run")
@mock.patch("lightning.app.core.api.UIRefresher")
@pytest.mark.parametrize("host", ["http://0.0.0.1", "0.0.0.1"])
def test_start_server_info_message(ui_refresher, uvicorn_run, caplog, monkeypatch, host):
    api_publish_state_queue = _MockQueue()
    api_delta_queue = _MockQueue()
    has_started_queue = _MockQueue()
    api_response_queue = _MockQueue()
    kwargs = {
        "host": host,
        "port": 1111,
        "api_publish_state_queue": api_publish_state_queue,
        "api_delta_queue": api_delta_queue,
        "has_started_queue": has_started_queue,
        "api_response_queue": api_response_queue,
        "root_path": "test",
    }

    monkeypatch.setattr(api, "logger", logging.getLogger())

    with caplog.at_level(logging.INFO):
        start_server(**kwargs)

    assert "Your app has started. View it in your browser: http://0.0.0.1:1111/view" in caplog.text

    ui_refresher.assert_called_once()
    uvicorn_run.assert_called_once_with(host="0.0.0.1", port=1111, log_level="error", app=mock.ANY, root_path="test")


class InputRequestModel(BaseModel):
    index: int
    name: str


class OutputRequestModel(BaseModel):
    name: str
    counter: int


async def handler():
    print("Has been called")
    return "Hello World !"


class FlowAPI(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        if self.counter == 501:
            self.stop()

    def request(self, config: InputRequestModel, request: Request) -> OutputRequestModel:
        self.counter += 1
        if config.index % 5 == 0:
            raise HTTPException(status_code=400, detail="HERE")
        assert request.body()
        assert request.json()
        assert request.headers
        assert request.method
        return OutputRequestModel(name=config.name, counter=self.counter)

    def configure_api(self):
        return [Post("/api/v1/request", self.request), Post("/api/v1/handler", handler)]


def target():
    app = LightningApp(FlowAPI())
    MultiProcessRuntime(app).dispatch()


async def async_request(url: str, data: InputRequestModel):
    async with aiohttp.ClientSession() as session, session.post(url, json=data.dict()) as result:
        return await result.json()


@pytest.mark.xfail(strict=False, reason="No idea why... need to be fixed")  # fixme
def test_configure_api():
    # Setup
    process = Process(target=target)
    process.start()
    time_left = 15
    while time_left > 0:
        try:
            requests.get(f"http://localhost:{APP_SERVER_PORT}/healthz")
            break
        except requests.exceptions.ConnectionError:
            sleep(0.1)
            time_left -= 0.1

    # Test Upload File
    with open(__file__, "rb") as fo:
        files = {"uploaded_file": fo}

    response = requests.put(f"http://localhost:{APP_SERVER_PORT}/api/v1/upload_file/test", files=files)
    assert response.json() == "Successfully uploaded 'test' to the Drive"

    url = f"http://localhost:{APP_SERVER_PORT}/api/v1/request"

    N = 500
    coros = []
    for index in range(N):
        coros.append(async_request(url, InputRequestModel(index=index, name="hello")))

    t0 = time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(asyncio.gather(*coros))
    response_time = time() - t0
    print(f"RPS: {N / response_time}")
    assert response_time < 10
    assert len(results) == N
    assert all(r.get("detail", None) == ("HERE" if i % 5 == 0 else None) for i, r in enumerate(results))

    response = requests.post(f"http://localhost:{APP_SERVER_PORT}/api/v1/handler")
    assert response.status_code == 200

    # Stop the Application
    with contextlib.suppress(Exception):
        response = requests.post(url, json=InputRequestModel(index=0, name="hello").dict())

    # Teardown
    time_left = 5
    while time_left > 0:
        if process.exitcode == 0:
            break
        sleep(0.1)
        time_left -= 0.1
    assert process.exitcode == 0
    process.kill()


@pytest.mark.anyio()
@mock.patch("lightning.app.core.api.UIRefresher", mock.MagicMock())
async def test_get_annotations(tmpdir):
    cwd = os.getcwd()
    os.chdir(tmpdir)

    Path("lightning-annotations.json").write_text('[{"test": 3}]')

    try:
        app = AppStageTestingApp(FlowA(), log_level="debug")
        app._update_layout()
        app.stage = AppStage.BLOCKING
        change_state_queue = _MockQueue("change_state_queue")
        has_started_queue = _MockQueue("has_started_queue")
        api_response_queue = _MockQueue("api_response_queue")
        spec = extract_metadata_from_app(app)
        start_server(
            None,
            change_state_queue,
            api_response_queue,
            has_started_queue=has_started_queue,
            uvicorn_run=False,
            spec=spec,
        )

        async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
            response = await client.get("/api/v1/annotations")
            assert response.json() == [{"test": 3}]
    finally:
        # Cleanup
        os.chdir(cwd)
