import contextlib
from queue import Empty
from re import escape
from unittest.mock import MagicMock, Mock

import pytest
from lightning.app import LightningApp
from lightning.app.core.flow import LightningFlow
from lightning.app.core.work import LightningWork
from lightning.app.runners import MultiProcessRuntime
from lightning.app.storage.path import Path
from lightning.app.testing.helpers import EmptyFlow, EmptyWork, _MockQueue
from lightning.app.testing.testing import LightningTestApp
from lightning.app.utilities.enum import WorkStageStatus, make_status
from lightning.app.utilities.exceptions import LightningWorkException
from lightning.app.utilities.imports import _IS_WINDOWS
from lightning.app.utilities.packaging.build_config import BuildConfig
from lightning.app.utilities.proxies import ProxyWorkRun, WorkRunner


def test_lightning_work_run_method_required():
    """Test that a helpful exception is raised when the user did not implement the `LightningWork.run()` method."""
    with pytest.raises(TypeError, match=escape("The work `LightningWork` is missing the `run()` method")):
        LightningWork()

    class WorkWithoutRun(LightningWork):
        def __init__(self):
            super().__init__()
            self.started = False

    with pytest.raises(TypeError, match=escape("The work `WorkWithoutRun` is missing the `run()` method")):
        WorkWithoutRun()

    class WorkWithRun(WorkWithoutRun):
        def run(self, *args, **kwargs):
            self.started = True

    work = WorkWithRun()
    work.run()
    assert work.started


def test_lightning_work_no_children_allowed():
    """Test that a LightningWork can't have any children (work or flow)."""

    class ChildWork(EmptyWork):
        pass

    class ParentWork(LightningWork):
        def __init__(self):
            super().__init__()
            self.work_b = ChildWork()

        def run(self, *args, **kwargs):
            pass

    with pytest.raises(LightningWorkException, match="isn't allowed to take any children such as"):
        ParentWork()

    class ParentWork(LightningWork):
        def __init__(self):
            super().__init__()
            self.flow = LightningFlow()

        def run(self, *args, **kwargs):
            pass

    with pytest.raises(LightningWorkException, match="LightningFlow"):
        ParentWork()


def test_forgot_to_call_init():
    """This test validates the error message for user registering state without calling __init__ is comprehensible."""

    class W(LightningWork):
        def __init__(self):
            self.var_a = None

        def run(self):
            pass

    with pytest.raises(AttributeError, match="Did you forget to call"):
        W()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("x", 1),
        ("f", EmptyFlow()),
        ("w", EmptyWork()),
        ("run", lambda _: _),
    ],
)
def test_unsupported_attribute_declaration_outside_init(name, value):
    """Test that LightningWork attributes (with a few exceptions) are not allowed to be set outside __init__."""
    flow = EmptyFlow()
    with pytest.raises(AttributeError, match=f"Cannot set attributes that were not defined in __init__: {name}"):
        setattr(flow, name, value)
    assert name == "run" or not hasattr(flow, name)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("_name", "name"),
        ("_changes", {"change": 1}),
        ("run", ProxyWorkRun(work_run=Mock(), work_name="any", work=Mock(), caller_queue=Mock())),
    ],
)
def test_supported_attribute_declaration_outside_init(name, value):
    """Test the custom LightningWork setattr implementation for the few reserved attributes that are allowed to be set
    from outside __init__."""
    flow = EmptyWork()
    setattr(flow, name, value)
    assert getattr(flow, name) == value


def test_supported_attribute_declaration_inside_init():
    """Test that the custom LightningWork setattr can identify the __init__ call in the stack frames above."""

    class Work(EmptyWork):
        def __init__(self):
            super().__init__()
            self.directly_in_init = "init"
            self.method_under_init()

        def method_under_init(self):
            self.attribute = "test"

    work = Work()
    assert work.directly_in_init == "init"
    assert work.attribute == "test"


@pytest.mark.parametrize("replacement", [EmptyFlow(), EmptyWork(), None])
def test_fixing_flows_and_works(replacement):
    class FlowFixed(LightningFlow):
        def run(self):
            self.empty_flow = EmptyFlow()
            self.empty_flow = replacement

    with pytest.raises(AttributeError, match="Cannot set attributes as"):
        FlowFixed().run()


@pytest.mark.parametrize("enable_exception", [False, True])
@pytest.mark.parametrize("raise_exception", [False, True])
def test_lightning_status(enable_exception, raise_exception):
    class Work(EmptyWork):
        def __init__(self, raise_exception, enable_exception=True):
            super().__init__(raise_exception=raise_exception)
            self.enable_exception = enable_exception
            self.dummy_path = Path("test")

        def run(self):
            if self.enable_exception:
                raise Exception("Custom Exception")

    work = Work(raise_exception, enable_exception=enable_exception)
    work._name = "root.w"
    assert work.status.stage == WorkStageStatus.NOT_STARTED
    caller_queue = _MockQueue("caller_queue")
    delta_queue = _MockQueue("delta_queue")
    readiness_queue = _MockQueue("readiness_queue")
    error_queue = _MockQueue("error_queue")
    request_queue = _MockQueue("request_queue")
    response_queue = _MockQueue("response_queue")
    copy_request_queue = _MockQueue("copy_request_queue")
    copy_response_queue = _MockQueue("copy_response_queue")
    call_hash = "fe3fa0f"
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
    with contextlib.suppress(Exception, Empty):
        work_runner()

    res = delta_queue._queue[0].delta.to_dict()["iterable_item_added"]
    L = len(delta_queue._queue) - 1
    if enable_exception:
        exception_cls = Exception if raise_exception else Empty
        assert isinstance(error_queue._queue[0], exception_cls)
        res_end = delta_queue._queue[L].delta.to_dict()["iterable_item_added"]
        res_end[f"root['calls']['{call_hash}']['statuses'][1]"]["stage"] == "failed"
        res_end[f"root['calls']['{call_hash}']['statuses'][1]"]["message"] == "Custom Exception"
    else:
        assert res[f"root['calls']['{call_hash}']['statuses'][0]"]["stage"] == "running"
        key = f"root['calls']['{call_hash}']['statuses'][1]"
        while L >= 0:
            res_end = delta_queue._queue[L].delta.to_dict()["iterable_item_added"]
            if key in res_end and res_end[key]["stage"] == "succeeded":
                break
            L -= 1

    # Stop blocking and let the thread join
    work_runner.copier.join()


def test_lightning_work_url():
    class ExposedWork(LightningWork):
        def run(self):
            pass

    work = ExposedWork(port=8000)
    work._name = "root.work"
    assert work.state["vars"]["_url"] == ""


def test_work_path_assignment():
    """Test that paths in the lit format lit:// get converted to a proper lightning.app.storage.Path object."""

    class Work(LightningWork):
        def __init__(self):
            super().__init__()
            self.no_path = "a/b/c"
            self.path = Path("lit://x/y/z")
            self.lit_path = "lit://x/y/z"

        def run(self):
            pass

    work = Work()
    assert isinstance(work.no_path, str)
    assert isinstance(work.path, Path)
    assert isinstance(work.lit_path, Path)
    assert work.path == work.lit_path


@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut exception")
@pytest.mark.xfail(strict=False, reason="Timeout")  # fixme
def test_work_state_change_with_path():
    """Test that type changes to a Path attribute are properly reflected within the state."""

    class Work(LightningFlow):
        def __init__(self):
            super().__init__()
            self.none_to_path = None
            self.path_to_none = Path()
            self.path_to_path = Path()

        def run(self):
            self.none_to_path = "lit://none/to/path"
            self.path_to_none = None
            self.path_to_path = "lit://path/to/path"

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.work = Work()

        def run(self):
            self.work.run()
            self.stop()

    flow = Flow()
    MultiProcessRuntime(LightningApp(flow)).dispatch()
    assert flow.work.none_to_path == Path("lit://none/to/path")
    assert flow.work.path_to_none is None
    assert flow.work.path_to_path == Path("lit://path/to/path")

    assert "path_to_none" not in flow.work._paths
    assert "path_to_none" in flow.work._state
    assert flow.work._paths["none_to_path"] == Path("lit://none/to/path").to_dict()
    assert flow.work._paths["path_to_path"] == Path("lit://path/to/path").to_dict()
    assert flow.work.state["vars"]["none_to_path"] == Path("lit://none/to/path")
    assert flow.work.state["vars"]["path_to_none"] is None
    assert flow.work.state["vars"]["path_to_path"] == Path("lit://path/to/path")


def test_lightning_work_calls():
    class W(LightningWork):
        def run(self, *args, **kwargs):
            pass

    w = W()
    assert len(w._calls) == 1
    w.run(1, [2], (3, 4), {"1": "3"})
    assert len(w._calls) == 2
    assert w._calls["0d824f7"] == {"ret": None}


def test_work_cloud_build_config_provided():
    assert isinstance(LightningWork.cloud_build_config, property)
    assert LightningWork.cloud_build_config.fset is not None

    class Work(LightningWork):
        def __init__(self):
            super().__init__()
            self.cloud_build_config = BuildConfig(image="ghcr.io/gridai/base-images:v1.8-cpu")

        def run(self, *args, **kwargs):
            pass

    w = Work()
    w.run()


def test_work_local_build_config_provided():
    assert isinstance(LightningWork.local_build_config, property)
    assert LightningWork.local_build_config.fset is not None

    class Work(LightningWork):
        def __init__(self):
            super().__init__()
            self.local_build_config = BuildConfig(image="ghcr.io/gridai/base-images:v1.8-cpu")

        def run(self, *args, **kwargs):
            pass

    w = Work()
    w.run()


class WorkCounter(LightningWork):
    def run(self):
        pass


class LightningTestAppWithWork(LightningTestApp):
    def on_before_run_once(self):
        if self.root.work.has_succeeded:
            return True
        return super().on_before_run_once()


def test_lightning_app_with_work():
    app = LightningTestAppWithWork(WorkCounter())
    MultiProcessRuntime(app, start_server=False).dispatch()


class WorkStart(LightningWork):
    def __init__(self, cache_calls, parallel):
        super().__init__(cache_calls=cache_calls, parallel=parallel)
        self.counter = 0

    def run(self):
        self.counter += 1


class FlowStart(LightningFlow):
    def __init__(self, cache_calls, parallel):
        super().__init__()
        self.w = WorkStart(cache_calls, parallel)
        self.finish = False

    def run(self):
        if self.finish:
            self.stop()
        if self.w.status.stage == WorkStageStatus.STOPPED:
            with pytest.raises(Exception, match="A work can be started only once for now."):
                self.w.start()
            self.finish = True
        if self.w.status.stage == WorkStageStatus.NOT_STARTED:
            self.w.start()
        if self.w.status.stage == WorkStageStatus.STARTED:
            self.w.run()
        if self.w.counter == 1:
            self.w.stop()


@pytest.mark.parametrize("cache_calls", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_lightning_app_work_start(cache_calls, parallel):
    app = LightningApp(FlowStart(cache_calls, parallel))
    MultiProcessRuntime(app, start_server=False).dispatch()


def test_lightning_work_delete():
    work = WorkCounter()

    with pytest.raises(Exception, match="Can't delete the work"):
        work.delete()

    mock = MagicMock()
    work._backend = mock
    work.delete()
    assert work == mock.delete_work._mock_call_args_list[0].args[1]


class WorkDisplay(LightningWork):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


def test_lightning_work_display_name():
    work = WorkDisplay()
    assert work.state_vars["vars"]["_display_name"] == ""
    work.display_name = "Hello"
    assert work.state_vars["vars"]["_display_name"] == "Hello"

    work._calls["latest_call_hash"] = "test"
    work._calls["test"] = {"statuses": [make_status(WorkStageStatus.PENDING)]}
    with pytest.raises(RuntimeError, match="The display name can be set only before the work has started."):
        work.display_name = "HELLO"
    work.display_name = "Hello"
