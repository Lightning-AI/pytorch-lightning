import logging
import multiprocessing as mp
import os
from copy import deepcopy
from unittest import mock

import pytest
from deepdiff import DeepDiff, Delta
from httpx import AsyncClient

from lightning_app import LightningApp, LightningFlow, LightningWork
from lightning_app.core import api
from lightning_app.core.api import fastapi_service, global_app_state_store, start_server, UIRefresher
from lightning_app.runners import MultiProcessRuntime, SingleProcessRuntime
from lightning_app.storage.drive import Drive
from lightning_app.testing.helpers import MockQueue
from lightning_app.utilities.component import _set_frontend_context, _set_work_context
from lightning_app.utilities.enum import AppStage
from lightning_app.utilities.load_app import extract_metadata_from_app
from lightning_app.utilities.redis import check_if_redis_running
from lightning_app.utilities.state import AppState, headers_for


class WorkA(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
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
            self._exit()
        self.work_a.run()


# TODO: Resolve singleprocess - idea: explore frame calls recursively.
@pytest.mark.parametrize("runtime_cls", [MultiProcessRuntime])
def test_app_state_api(runtime_cls):
    """This test validates the AppState can properly broadcast changes from work within its own process."""
    app = LightningApp(_A())
    runtime_cls(app, start_server=True).dispatch()
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
            self._exit()


# TODO: Find why this test is flaky.
@pytest.mark.skipif(True, reason="flaky test.")
@pytest.mark.parametrize("runtime_cls", [SingleProcessRuntime])
def test_app_state_api_with_flows(runtime_cls, tmpdir):
    """This test validates the AppState can properly broadcast changes from flows."""
    app = LightningApp(A2(), debug=True)
    runtime_cls(app, start_server=True).dispatch()
    assert app.root.var_a == -1


class FlowA(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        self.counter += 1
        if self.counter >= 3:
            self._exit()


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
        deep_diff = DeepDiff(previous_state, current_state)
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
@pytest.mark.skipif(True, reason="TODO: Resolve flaky test.")
@pytest.mark.parametrize("runtime_cls", [SingleProcessRuntime, MultiProcessRuntime])
def test_app_stage_from_frontend(runtime_cls):
    """This test validates that delta from the `api_delta_queue` manipulating the ['app_state']['stage'] would start and
    stop the app."""
    app = AppStageTestingApp(FlowA(), debug=True)
    app.stage = AppStage.BLOCKING
    runtime_cls(app, start_server=True).dispatch()


def test_update_publish_state_and_maybe_refresh_ui():
    """This test checks that the method properly:

    - receives the state from the `publish_state_queue` and populates the app_state_store
    - receives a notification to refresh the UI and makes a GET Request (streamlit).
    """

    app = AppStageTestingApp(FlowA(), debug=True)
    publish_state_queue = MockQueue("publish_state_queue")

    publish_state_queue.put(app.state_with_changes)

    thread = UIRefresher(publish_state_queue)
    thread.run_once()

    assert global_app_state_store.get_app_state("1234") == app.state_with_changes
    global_app_state_store.remove("1234")
    global_app_state_store.add("1234")


@pytest.mark.parametrize("x_lightning_type", ["DEFAULT", "STREAMLIT"])
@pytest.mark.anyio
async def test_start_server(x_lightning_type):
    """This test relies on FastAPI TestClient and validates that the REST API properly provides:

    - the state on GET /api/v1/state
    - push a delta when making a POST request to /api/v1/state
    """

    class InfiniteQueue(MockQueue):
        def get(self, timeout: int = 0):
            return self._queue[0]

    app = AppStageTestingApp(FlowA(), debug=True)
    app.stage = AppStage.BLOCKING
    publish_state_queue = InfiniteQueue("publish_state_queue")
    change_state_queue = MockQueue("change_state_queue")
    has_started_queue = MockQueue("has_started_queue")
    state = app.state_with_changes
    publish_state_queue.put(state)
    spec = extract_metadata_from_app(app)
    ui_refresher = start_server(
        publish_state_queue, change_state_queue, has_started_queue=has_started_queue, uvicorn_run=False, spec=spec
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

        # used to clean the app_state_store to following test.
        global_app_state_store.remove("1234")
        global_app_state_store.add("1234")

    del client
    ui_refresher.join(0)


@pytest.mark.parametrize(
    "path, expected_status_code",
    (
        ("/api/v1", 404),
        ("/api/v1/asdf", 404),
        ("/api/asdf", 404),
        ("/api", 404),
    ),
)
@pytest.mark.anyio
async def test_state_api_routes(path, expected_status_code):
    async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == expected_status_code


@pytest.mark.skipif(not check_if_redis_running(), reason="redis not running")
@pytest.mark.anyio
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
@pytest.mark.anyio
async def test_health_endpoint_failure():
    async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
        # will respond 503 if redis is not running
        response = await client.get("/healthz")
        assert response.status_code == 500


@pytest.mark.parametrize(
    "path, expected_status_code",
    (
        ("/", 200),
        ("/asdf", 200),
        ("/view/component_a", 200),
        ("/admin", 200),
    ),
)
@pytest.mark.anyio
async def test_frontend_routes(path, expected_status_code):
    async with AsyncClient(app=fastapi_service, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == expected_status_code


def test_start_server_started():
    """This test ensures has_started_queue receives a signal when the REST API has started."""
    api_publish_state_queue = mp.Queue()
    api_delta_queue = mp.Queue()
    has_started_queue = mp.Queue()
    kwargs = dict(
        api_publish_state_queue=api_publish_state_queue,
        api_delta_queue=api_delta_queue,
        has_started_queue=has_started_queue,
        port=1111,
    )

    server_proc = mp.Process(target=start_server, kwargs=kwargs)
    server_proc.start()
    # requires to wait for the UI to be clicked on.

    # wait for server to be ready
    assert has_started_queue.get() == "SERVER_HAS_STARTED"
    server_proc.kill()


@mock.patch("uvicorn.run")
@mock.patch("lightning_app.core.api.UIRefresher")
@pytest.mark.parametrize("host", ["http://0.0.0.1", "0.0.0.1"])
def test_start_server_info_message(ui_refresher, uvicorn_run, caplog, monkeypatch, host):
    api_publish_state_queue = MockQueue()
    api_delta_queue = MockQueue()
    has_started_queue = MockQueue()
    kwargs = dict(
        host=host,
        port=1111,
        api_publish_state_queue=api_publish_state_queue,
        api_delta_queue=api_delta_queue,
        has_started_queue=has_started_queue,
    )

    monkeypatch.setattr(api, "logger", logging.getLogger())

    with caplog.at_level(logging.INFO):
        start_server(**kwargs)

    assert "Your app has started. View it in your browser: http://0.0.0.1:1111/view" in caplog.text

    ui_refresher.assert_called_once()
    uvicorn_run.assert_called_once_with(host="0.0.0.1", port=1111, log_level="error", app=mock.ANY)
