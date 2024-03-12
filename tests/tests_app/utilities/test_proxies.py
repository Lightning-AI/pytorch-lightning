import contextlib
import logging
import os
import pathlib
import sys
import time
import traceback
from copy import deepcopy
from queue import Empty
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
from deepdiff import DeepDiff, Delta
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.runners import MultiProcessRuntime
from lightning.app.storage import Drive
from lightning.app.storage.path import Path, _artifacts_path
from lightning.app.storage.requests import _GetRequest
from lightning.app.testing.helpers import EmptyFlow, _MockQueue
from lightning.app.utilities.component import _convert_paths_after_init
from lightning.app.utilities.enum import AppStage, CacheCallsKeys, WorkFailureReasons, WorkStageStatus
from lightning.app.utilities.exceptions import CacheMissException, ExitAppException
from lightning.app.utilities.proxies import (
    ComponentDelta,
    LightningWorkSetAttrProxy,
    ProxyWorkRun,
    WorkRunner,
    WorkStateObserver,
    persist_artifacts,
)

logger = logging.getLogger(__name__)


class Work(LightningWork):
    def __init__(self, cache_calls=True, parallel=True):
        super().__init__(cache_calls=cache_calls, parallel=parallel)
        self.counter = 0

    def run(self):
        self.counter = 1
        return 1


def test_lightning_work_setattr():
    """This test valides that the `LightningWorkSetAttrProxy` would push a delta to the `caller_queue` everytime an
    attribute from the work state is being changed."""

    w = Work()
    # prepare
    w._name = "root.b"
    # create queue
    caller_queue = _MockQueue("caller_queue")

    def proxy_setattr():
        w._setattr_replacement = LightningWorkSetAttrProxy(w._name, w, caller_queue, MagicMock())

    proxy_setattr()
    w.run()
    assert len(caller_queue) == 1
    work_proxy_output = caller_queue._queue[0]
    assert isinstance(work_proxy_output, ComponentDelta)
    assert work_proxy_output.id == w._name
    assert work_proxy_output.delta.to_dict() == {"values_changed": {"root['vars']['counter']": {"new_value": 1}}}


@pytest.mark.parametrize(
    ("parallel", "cache_calls"),
    [
        (True, True),
        (True, False),
        (False, True),
        pytest.param(False, False, marks=pytest.mark.xfail(strict=False, reason="failing...")),  # fixme
    ],
)
@mock.patch("lightning.app.utilities.proxies._Copier", MagicMock())
@pytest.mark.flaky(reruns=3)
@pytest.mark.xfail(sys.platform == "win32", strict=False, reason="Fix this on Windows")  # TODO @ethanwharris
def test_work_runner(parallel, cache_calls, *_):
    """This test validates the `WorkRunner` runs the work.run method and properly populates the `delta_queue`,
    `error_queue` and `readiness_queue`."""

    class Work(LightningWork):
        def __init__(self, cache_calls=True, parallel=True):
            super().__init__(cache_calls=cache_calls, parallel=parallel)
            self.counter = 0
            self.dummy_path = "lit://test"

        def run(self):
            self.counter = 1

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.w = Work(cache_calls=cache_calls, parallel=parallel)

        def run(self):
            pass

    class BlockingQueue(_MockQueue):
        """A Mock for the file copier queues that keeps blocking until we want to end the thread."""

        keep_blocking = True

        def get(self, timeout: int = 0):
            while BlockingQueue.keep_blocking:
                pass
            # A dummy request so the Copier gets something to process without an error
            return _GetRequest(source="src", name="dummy_path", path="test", hash="123", destination="dst")

    app = LightningApp(Flow())
    work = app.root.w
    caller_queue = _MockQueue("caller_queue")
    delta_queue = _MockQueue("delta_queue")
    readiness_queue = _MockQueue("readiness_queue")
    error_queue = _MockQueue("error_queue")
    request_queue = _MockQueue("request_queue")
    response_queue = _MockQueue("response_queue")
    copy_request_queue = BlockingQueue("copy_request_queue")
    copy_response_queue = BlockingQueue("copy_response_queue")

    call_hash = "run:fe3fa0f34fc1317e152e5afb023332995392071046f1ea51c34c7c9766e3676c"
    work._calls[call_hash] = {
        "args": (),
        "kwargs": {},
        "call_hash": call_hash,
        "run_started_counter": 1,
        "statuses": [],
    }
    caller_queue.put({
        "args": (),
        "kwargs": {},
        "call_hash": call_hash,
        "state": work.state,
    })
    work_runner = WorkRunner(
        work,
        work.name,
        caller_queue,
        delta_queue,
        readiness_queue,
        error_queue,
        request_queue,
        response_queue,
        copy_request_queue,
        copy_response_queue,
    )
    with contextlib.suppress(Empty, Exception):
        work_runner()

    assert readiness_queue._queue[0]
    if parallel:
        assert isinstance(error_queue._queue[0], Exception)
    else:
        assert isinstance(error_queue._queue[0], Empty)
        assert len(delta_queue._queue) in [3, 4]
        res = delta_queue._queue[0].delta.to_dict()["iterable_item_added"]
        assert res[f"root['calls']['{call_hash}']['statuses'][0]"]["stage"] == "running"
        assert delta_queue._queue[1].delta.to_dict() == {
            "values_changed": {"root['vars']['counter']": {"new_value": 1}}
        }
        index = 3 if len(delta_queue._queue) == 4 else 2
        res = delta_queue._queue[index].delta.to_dict()["dictionary_item_added"]
        assert res[f"root['calls']['{call_hash}']['ret']"] is None

    # Stop blocking and let the thread join
    BlockingQueue.keep_blocking = False
    work_runner.copier.join()


def test_pathlike_as_argument_to_run_method_warns(tmpdir):
    """Test that Lightning Produces a special warning for strings that look like paths."""
    # all these paths are not proper paths or don't have a file or folder that exists
    no_warning_expected = (
        "looks/like/path",
        pathlib.Path("looks/like/path"),
        "i am not a path",
        1,
        Path("lightning/path"),
    )
    for path in no_warning_expected:
        _pass_path_argument_to_work_and_test_warning(path=path, warning_expected=False)

    # warn if it looks like a folder and the folder exists
    _pass_path_argument_to_work_and_test_warning(path=tmpdir, warning_expected=True)

    # warn if it looks like a string or pathlib Path and the file exists
    file = pathlib.Path(tmpdir, "file_exists.txt")
    file.write_text("test")
    assert os.path.exists(file)
    _pass_path_argument_to_work_and_test_warning(path=file, warning_expected=True)
    _pass_path_argument_to_work_and_test_warning(path=str(file), warning_expected=True)

    # do not warn if the path is wrapped in Lightning Path (and the file exists)
    file = Path(tmpdir, "file_exists.txt")
    file.write_text("test")
    assert os.path.exists(file)
    _pass_path_argument_to_work_and_test_warning(path=file, warning_expected=False)


def _pass_path_argument_to_work_and_test_warning(path, warning_expected):
    class WarnRunPathWork(LightningWork):
        def run(self, *args, **kwargs):
            pass

    class Flow(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.work = WarnRunPathWork()

    flow = Flow()
    work = flow.work
    proxy_run = ProxyWorkRun(work.run, "some", work, Mock())

    warn_ctx = pytest.warns(UserWarning, match="You passed a the value") if warning_expected else pytest.warns(None)
    with warn_ctx as record, pytest.raises(CacheMissException):
        proxy_run(path)

    assert warning_expected or all("You passed a the value" not in str(msg.message) for msg in record)


class WorkTimeout(LightningWork):
    def __init__(self):
        super().__init__(parallel=True, start_with_flow=False)
        self.counter = 0

    def run(self):
        self.counter += 1


class FlowTimeout(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.work = WorkTimeout()

    def run(self):
        if not self.work.has_started:
            self.work.run()
        if self.work.has_timeout:
            self.stop()


class WorkRunnerPatch(WorkRunner):
    counter = 0

    def __call__(self):
        call_hash = "fe3fa0f"
        while True:
            try:
                called = self.caller_queue.get()
                self.work.set_state(called["state"])
                state = deepcopy(self.work.state)
                self.work._calls[call_hash]["statuses"].append({
                    "name": self.work.name,
                    "stage": WorkStageStatus.FAILED,
                    "reason": WorkFailureReasons.TIMEOUT,
                    "timestamp": time.time(),
                    "message": None,
                })
                self.delta_queue.put(
                    ComponentDelta(id=self.work_name, delta=Delta(DeepDiff(state, self.work.state, verbose_level=2)))
                )
                self.counter += 1
            except Exception as ex:
                logger.error(traceback.format_exc())
                self.error_queue.put(ex)
                raise ExitAppException


@mock.patch("lightning.app.runners.backends.mp_process.WorkRunner", WorkRunnerPatch)
def test_proxy_timeout():
    app = LightningApp(FlowTimeout(), log_level="debug")
    MultiProcessRuntime(app, start_server=False).dispatch()

    call_hash = app.root.work._calls[CacheCallsKeys.LATEST_CALL_HASH]
    assert len(app.root.work._calls[call_hash]["statuses"]) == 3
    assert app.root.work._calls[call_hash]["statuses"][0]["stage"] == "pending"
    assert app.root.work._calls[call_hash]["statuses"][1]["stage"] == "failed"
    assert app.root.work._calls[call_hash]["statuses"][2]["stage"] == "stopped"


@mock.patch("lightning.app.utilities.proxies._Copier")
def test_path_argument_to_transfer(*_):
    """Test that any Lightning Path objects passed to the run method get transferred automatically (if they exist)."""

    class TransferPathWork(LightningWork):
        def run(self, *args, **kwargs):
            raise ExitAppException

    work = TransferPathWork()

    path1 = Path("exists-locally.txt")
    path2 = Path("exists-remotely.txt")
    path3 = Path("exists-nowhere.txt")

    path1.get = Mock()
    path2.get = Mock()
    path3.get = Mock()

    path1.exists_remote = Mock(return_value=False)
    path2.exists_remote = Mock(return_value=True)
    path3.exists_remote = Mock(return_value=False)

    path1._origin = "origin"
    path2._origin = "origin"
    path3._origin = "origin"

    call = {
        "args": (path1, path2),
        "kwargs": {"path3": path3},
        "call_hash": "any",
        "state": {
            "vars": {"_paths": {}, "_urls": {}},
            "calls": {
                CacheCallsKeys.LATEST_CALL_HASH: "any",
                "any": {
                    "name": "run",
                    "call_hash": "any",
                    "use_args": False,
                    "statuses": [{"stage": "requesting", "message": None, "reason": None, "timestamp": 1}],
                },
            },
            "changes": {},
        },
    }

    caller_queue = _MockQueue()
    caller_queue.put(call)

    runner = WorkRunner(
        work=work,
        work_name="name",
        caller_queue=caller_queue,
        delta_queue=_MockQueue(),
        readiness_queue=_MockQueue(),
        error_queue=_MockQueue(),
        request_queue=_MockQueue(),
        response_queue=_MockQueue(),
        copy_request_queue=_MockQueue(),
        copy_response_queue=_MockQueue(),
    )

    with contextlib.suppress(ExitAppException):
        runner()

    path1.exists_remote.assert_called_once()
    path1.get.assert_not_called()

    path2.exists_remote.assert_called_once()
    path2.get.assert_called_once()

    path3.exists_remote.assert_called()
    path3.get.assert_not_called()


@pytest.mark.parametrize(
    ("origin", "exists_remote", "expected_get"),
    [
        (None, False, False),
        ("root.work", True, False),
        ("root.work", False, False),
        ("origin", True, True),
    ],
)
@mock.patch("lightning.app.utilities.proxies._Copier")
def test_path_attributes_to_transfer(_, origin, exists_remote, expected_get):
    """Test that any Lightning Path objects passed to the run method get transferred automatically (if they exist)."""
    path_mock = Mock()
    path_mock.origin_name = origin
    path_mock.exists_remote = Mock(return_value=exists_remote)

    class TransferPathWork(LightningWork):
        def __init__(self):
            super().__init__()
            self.path = Path("test-path.txt")

        def run(self):
            raise ExitAppException

        def __getattr__(self, item):
            if item == "path":
                return path_mock
            return super().__getattr__(item)

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.work = TransferPathWork()

        def run(self):
            self.work.run()

    flow = Flow()
    _convert_paths_after_init(flow)

    call = {
        "args": (),
        "kwargs": {},
        "call_hash": "any",
        "state": {
            "vars": {"_paths": flow.work._paths, "_urls": {}},
            "calls": {
                CacheCallsKeys.LATEST_CALL_HASH: "any",
                "any": {
                    "name": "run",
                    "call_hash": "any",
                    "use_args": False,
                    "statuses": [{"stage": "requesting", "message": None, "reason": None, "timestamp": 1}],
                },
            },
            "changes": {},
        },
    }

    caller_queue = _MockQueue()
    caller_queue.put(call)

    runner = WorkRunner(
        work=flow.work,
        work_name=flow.work.name,
        caller_queue=caller_queue,
        delta_queue=_MockQueue(),
        readiness_queue=_MockQueue(),
        error_queue=_MockQueue(),
        request_queue=_MockQueue(),
        response_queue=_MockQueue(),
        copy_request_queue=_MockQueue(),
        copy_response_queue=_MockQueue(),
    )
    with contextlib.suppress(ExitAppException):
        runner()

    assert path_mock.get.call_count == expected_get


def test_proxy_work_run_paths_replace_origin_lightning_work_by_their_name():
    class Work(LightningWork):
        def __init__(self):
            super().__init__(parallel=True)
            self.path = None

        def run(self, path):
            assert isinstance(path._origin, str)

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.w1 = Work()
            self.w = Work()

        def run(self):
            pass

    app = LightningApp(Flow())
    work = app.root.w
    caller_queue = _MockQueue("caller_queue")
    app.root.w1.path = Path(__file__)
    assert app.root.w1.path._origin == app.root.w1
    ProxyWorkRun(work.run, work.name, work, caller_queue)(path=app.root.w1.path)
    assert caller_queue._queue[0]["kwargs"]["path"]._origin == app.root.w1.name


def test_persist_artifacts(tmp_path):
    """Test that the `persist_artifacts` utility copies the artifacts that exist to the persistent storage."""

    class ArtifactWork(LightningWork):
        def __init__(self):
            super().__init__()
            self.file = None
            self.folder = None
            self.not_my_path = None
            self.not_exists = None

        def run(self):
            # single file
            self.file = Path(tmp_path, "file.txt")
            self.file.write_text("single file")
            # folder with files
            self.folder = Path(tmp_path, "folder")
            self.folder.mkdir()
            Path(tmp_path, "folder", "file1.txt").write_text("file 1")
            Path(tmp_path, "folder", "file2.txt").write_text("file 2")

            # simulate a Path that was synced to this Work from another Work
            self.not_my_path = Path(tmp_path, "external.txt")
            self.not_my_path.touch()
            self.not_my_path._origin = Mock()

            self.not_exists = Path(tmp_path, "not-exists")

    work = ArtifactWork()
    work._name = "root.work"

    rel_tmpdir_path = Path(*tmp_path.parts[1:])

    assert not os.path.exists(_artifacts_path(work) / rel_tmpdir_path / "file.txt")
    assert not os.path.exists(_artifacts_path(work) / rel_tmpdir_path / "folder")
    assert not os.path.exists(_artifacts_path(work) / rel_tmpdir_path / "not-exists")

    work.run()

    with pytest.warns(UserWarning, match="1 artifacts could not be saved because they don't exist"):
        persist_artifacts(work)

    assert os.path.exists(_artifacts_path(work) / rel_tmpdir_path / "file.txt")
    assert os.path.exists(_artifacts_path(work) / rel_tmpdir_path / "folder")
    assert not os.path.exists(_artifacts_path(work) / rel_tmpdir_path / "not-exists")
    assert not os.path.exists(_artifacts_path(work) / rel_tmpdir_path / "external.txt")


def test_work_state_observer():
    """Tests that the WorkStateObserver sends deltas to the queue when state residuals remain that haven't been handled
    by the setattr."""

    class WorkWithoutSetattr(LightningWork):
        def __init__(self):
            super().__init__()
            self.var = 1
            self.list = []
            self.dict = {"counter": 0}

        def run(self, use_setattr=False, use_containers=False):
            if use_setattr:
                self.var += 1
            if use_containers:
                self.list.append(1)
                self.dict["counter"] += 1

    work = WorkWithoutSetattr()
    delta_queue = _MockQueue()
    observer = WorkStateObserver(work, delta_queue)
    setattr_proxy = LightningWorkSetAttrProxy(
        work=work,
        work_name="work_name",
        delta_queue=delta_queue,
        state_observer=observer,
    )
    work._setattr_replacement = setattr_proxy

    ##############################
    # 1. Simulate no state changes
    ##############################
    work.run(use_setattr=False, use_containers=False)
    assert len(delta_queue) == 0

    ############################
    # 2. Simulate a setattr call
    ############################
    work.run(use_setattr=True, use_containers=False)

    # this is necessary only in this test where we simulate the calls
    work._calls.clear()
    work._calls.update({CacheCallsKeys.LATEST_CALL_HASH: None})

    delta = delta_queue.get().delta.to_dict()
    assert delta["values_changed"] == {"root['vars']['var']": {"new_value": 2}}
    assert len(observer._delta_memory) == 1

    # The observer should not trigger any deltas being sent and only consume the delta memory
    assert len(delta_queue) == 0
    observer.run_once()
    assert len(delta_queue) == 0
    assert not observer._delta_memory

    ################################
    # 3. Simulate a container update
    ################################
    work.run(use_setattr=False, use_containers=True)
    assert len(delta_queue) == 0
    assert not observer._delta_memory
    observer.run_once()
    observer.run_once()  # multiple runs should not affect how many deltas are sent unless there are changes
    delta = delta_queue.get().delta.to_dict()
    assert delta["values_changed"] == {"root['vars']['dict']['counter']": {"new_value": 1}}
    assert delta["iterable_item_added"] == {"root['vars']['list'][0]": 1}

    ##########################
    # 4. Simulate both updates
    ##########################
    work.run(use_setattr=True, use_containers=True)

    # this is necessary only in this test where we siumulate the calls
    work._calls.clear()
    work._calls.update({CacheCallsKeys.LATEST_CALL_HASH: None})

    delta = delta_queue.get().delta.to_dict()
    assert delta == {"values_changed": {"root['vars']['var']": {"new_value": 3}}}
    assert len(delta_queue) == 0
    assert len(observer._delta_memory) == 1
    observer.run_once()

    delta = delta_queue.get().delta.to_dict()
    assert delta["values_changed"] == {"root['vars']['dict']['counter']": {"new_value": 2}}
    assert delta["iterable_item_added"] == {"root['vars']['list'][1]": 1}

    assert len(delta_queue) == 0
    assert not observer._delta_memory


class WorkState(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.vars = []
        self.counter = 0

    def run(self, *args):
        for counter in range(1, 11):
            self.vars.append(counter)
            self.counter = counter


class FlowState(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = WorkState()
        self.counter = 1

    def run(self):
        self.w.run()
        if self.counter == 1:
            if len(self.w.vars) == 10 and self.w.counter == 10:
                self.w.vars = []
                self.w.counter = 0
                self.w.run("")
                self.counter = 2
        elif self.counter == 2 and len(self.w.vars) == 10 and self.w.counter == 10:
            self.stop()


def test_state_observer():
    app = LightningApp(FlowState())
    MultiProcessRuntime(app, start_server=False).dispatch()


@pytest.mark.parametrize(
    ("patch_constants", "environment", "expected_public_ip", "expected_private_ip"),
    [
        ({}, {}, "", "127.0.0.1"),
        ({"LIGHTNING_CLOUDSPACE_HOST": "any"}, {}, "", "0.0.0.0"),  # noqa: S104
        (
            {},
            {"LIGHTNING_NODE_IP": "85.44.2.25", "LIGHTNING_NODE_PRIVATE_IP": "10.10.10.5"},
            "85.44.2.25",
            "10.10.10.5",
        ),
    ],
    indirect=["patch_constants"],
)
def test_work_runner_sets_public_and_private_ip(patch_constants, environment, expected_public_ip, expected_private_ip):
    """Test that the WorkRunner updates the public and private address as soon as the Work starts running."""

    class Work(LightningWork):
        def run(self):
            pass

    work = Work()
    work_runner = WorkRunner(
        work,
        work.name,
        caller_queue=_MockQueue("caller_queue"),
        delta_queue=Mock(),
        readiness_queue=Mock(),
        error_queue=Mock(),
        request_queue=Mock(),
        response_queue=Mock(),
        copy_request_queue=Mock(),
        copy_response_queue=Mock(),
    )

    # Make a fake call
    call_hash = "run:fe3fa0f34fc1317e152e5afb023332995392071046f1ea51c34c7c9766e3676c"
    work._calls[call_hash] = {
        "args": (),
        "kwargs": {},
        "call_hash": call_hash,
        "run_started_counter": 1,
        "statuses": [],
    }
    work_runner.caller_queue.put({
        "args": (),
        "kwargs": {},
        "call_hash": call_hash,
        "state": work.state,
    })

    with mock.patch.dict(os.environ, environment, clear=True):
        work_runner.setup()
        # The public ip address only becomes available once the hardware is up / the work is running.
        assert work.public_ip == ""
        assert work.internal_ip == ""
        with contextlib.suppress(Empty):
            work_runner.run_once()
        assert work.public_ip == expected_public_ip
        assert work.internal_ip == expected_private_ip


class WorkBi(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.finished = False
        self.counter = 0
        self.counter_2 = 0

    def run(self):
        while not self.finished:
            self.counter_2 += 1
            time.sleep(0.1)
        self.counter = -1
        time.sleep(1)


class FlowBi(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = WorkBi()

    def run(self):
        self.w.run()
        if not self.w.finished:
            self.w.counter += 1
        if self.w.counter > 3:
            self.w.finished = True
        if self.w.counter == -1 and self.w.has_succeeded:
            self.stop()


def test_bi_directional_proxy():
    app = LightningApp(FlowBi())
    MultiProcessRuntime(app, start_server=False).dispatch()


class WorkBi2(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.finished = False
        self.counter = 0
        self.d = {}

    def run(self):
        self.counter -= 1
        while not self.finished:
            self.counter -= 1
            time.sleep(1)


class FlowBi2(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = WorkBi2()

    def run(self):
        self.w.run()
        if self.w.counter == 1:
            self.w.d["self.w.counter"] = 0
        if not self.w.finished:
            self.w.counter += 1


def test_bi_directional_proxy_forbidden(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(sys, "exit", mock)
    app = LightningApp(FlowBi2())
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.stage == AppStage.FAILED
    assert "A forbidden operation to update the work" in str(app.exception)


class WorkDrive(LightningFlow):
    def __init__(self, drive):
        super().__init__()
        self.drive = drive
        self.path = Path("data")

    def run(self):
        pass


class FlowDrive(LightningFlow):
    def __init__(self):
        super().__init__()
        self.data = Drive("lit://data")
        self.counter = 0

    def run(self):
        if not hasattr(self, "w"):
            self.w = WorkDrive(self.data)
            self.counter += 1


def test_bi_directional_proxy_filtering():
    app = LightningApp(FlowDrive())
    app.root.run()
    assert app._extract_vars_from_component_name(app.root.w.name, app.state) == {}
