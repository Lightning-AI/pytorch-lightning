from queue import Empty
from unittest.mock import Mock

import pytest

from lightning_app import LightningApp
from lightning_app.core.flow import LightningFlow
from lightning_app.core.work import LightningWork, LightningWorkException
from lightning_app.runners import MultiProcessRuntime
from lightning_app.storage import Path
from lightning_app.storage.requests import GetRequest
from lightning_app.testing.helpers import EmptyFlow, EmptyWork, MockQueue
from lightning_app.utilities.enum import WorkStageStatus
from lightning_app.utilities.proxies import ProxyWorkRun, WorkRunner


def test_simple_lightning_work():
    class Work_A(LightningWork):
        def __init__(self):
            super().__init__()
            self.started = False

    with pytest.raises(TypeError, match="Work_A"):
        Work_A()

    class Work_B(Work_A):
        def run(self, *args, **kwargs):
            self.started = True

    work_b = Work_B()
    work_b.run()
    assert work_b.started

    class Work_C(LightningWork):
        def __init__(self):
            super().__init__()
            self.work_b = Work_B()

        def run(self, *args, **kwargs):
            pass

    with pytest.raises(LightningWorkException, match="isn't allowed to take any children such as"):
        Work_C()

    class Work_C(LightningWork):
        def __init__(self):
            super().__init__()
            self.flow = LightningFlow()

        def run(self, *args, **kwargs):
            pass

    with pytest.raises(LightningWorkException, match="LightningFlow"):
        Work_C()


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
    "name,value",
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
    "name,value",
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


@pytest.mark.parametrize("raise_exception", [False, True])
@pytest.mark.parametrize("enable_exception", [False, True])
def test_lightning_status(enable_exception, raise_exception):
    class Work(EmptyWork):
        def __init__(self, raise_exception, enable_exception=True):
            super().__init__(raise_exception=raise_exception)
            self.enable_exception = enable_exception
            self.dummy_path = Path("test")

        def run(self):
            if self.enable_exception:
                raise Exception("Custom Exception")

    class BlockingQueue(MockQueue):
        """A Mock for the file copier queues that keeps blocking until we want to end the thread."""

        keep_blocking = True

        def get(self, timeout: int = 0):
            while BlockingQueue.keep_blocking:
                pass
            # A dummy request so the Copier gets something to process without an error
            return GetRequest(source="src", name="dummy_path", path="test", hash="123", destination="dst")

    work = Work(raise_exception, enable_exception=enable_exception)
    work._name = "root.w"
    assert work.status.stage == WorkStageStatus.NOT_STARTED
    caller_queue = MockQueue("caller_queue")
    delta_queue = MockQueue("delta_queue")
    readiness_queue = MockQueue("readiness_queue")
    error_queue = MockQueue("error_queue")
    request_queue = MockQueue("request_queue")
    response_queue = MockQueue("response_queue")
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
    caller_queue.put(
        {
            "args": (),
            "kwargs": {},
            "call_hash": call_hash,
            "state": work.state,
        }
    )
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
    try:
        work_runner()
    except (Exception, Empty):
        pass

    res = delta_queue._queue[0].delta.to_dict()["iterable_item_added"]
    res_end = delta_queue._queue[1].delta.to_dict()["iterable_item_added"]
    if enable_exception:
        exception_cls = Exception if raise_exception else Empty
        assert isinstance(error_queue._queue[0], exception_cls)
        res[f"root['calls']['{call_hash}']['statuses'][0]"]["stage"] == "failed"
        res[f"root['calls']['{call_hash}']['statuses'][0]"]["message"] == "Custom Exception"
    else:
        assert res[f"root['calls']['{call_hash}']['statuses'][0]"]["stage"] == "running"
        assert res_end[f"root['calls']['{call_hash}']['statuses'][1]"]["stage"] == "succeeded"

    # Stop blocking and let the thread join
    BlockingQueue.keep_blocking = False
    work_runner.copier.join()


def test_lightning_work_url():
    class ExposedWork(LightningWork):
        def run(self):
            pass

    work = ExposedWork(port=8000)
    work._name = "root.work"
    assert work.state["vars"]["_url"] == ""


def test_work_path_assignment():
    """Test that paths in the lit format lit:// get converted to a proper lightning_app.storage.Path object."""

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
            self._exit()

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
