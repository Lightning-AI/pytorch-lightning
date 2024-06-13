import contextlib
import logging
import os
import pickle
from re import escape
from time import sleep, time
from unittest import mock

import pytest
from deepdiff import Delta
from lightning.app import CloudCompute, LightningApp, LightningFlow, LightningWork  # F401
from lightning.app.api.request_types import _DeltaRequest
from lightning.app.core.constants import (
    FLOW_DURATION_SAMPLES,
    FLOW_DURATION_THRESHOLD,
    REDIS_QUEUES_READ_DEFAULT_TIMEOUT,
    STATE_UPDATE_TIMEOUT,
)
from lightning.app.core.queues import BaseQueue, MultiProcessQueue, RedisQueue
from lightning.app.frontend import StreamlitFrontend
from lightning.app.runners import MultiProcessRuntime
from lightning.app.storage.path import Path, _storage_root_dir
from lightning.app.testing.helpers import _RunIf
from lightning.app.testing.testing import LightningTestApp
from lightning.app.utilities.app_helpers import affiliation
from lightning.app.utilities.enum import AppStage, WorkStageStatus, WorkStopReasons
from lightning.app.utilities.imports import _IS_WINDOWS
from lightning.app.utilities.packaging import cloud_compute
from lightning.app.utilities.redis import check_if_redis_running
from lightning.app.utilities.warnings import LightningFlowWarning
from lightning_utilities.core.imports import RequirementCache
from pympler import asizeof

from tests_app import _PROJECT_ROOT

_STREAMLIT_AVAILABLE = RequirementCache("streamlit")

logger = logging.getLogger()


def test_lightning_app_requires_root_run_method():
    """Test that a useful exception is raised if the root flow does not override the run method."""
    with pytest.raises(
        TypeError, match=escape("The root flow passed to `LightningApp` does not override the `run()` method")
    ):
        LightningApp(LightningFlow())

    class FlowWithoutRun(LightningFlow):
        pass

    with pytest.raises(
        TypeError, match=escape("The root flow passed to `LightningApp` does not override the `run()` method")
    ):
        LightningApp(FlowWithoutRun())

    class FlowWithRun(LightningFlow):
        def run(self):
            pass

    LightningApp(FlowWithRun())  # no error


class B1(LightningFlow):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class A1(LightningFlow):
    def __init__(self):
        super().__init__()
        self.b = B1()

    def run(self):
        pass


class Work(LightningWork):
    def __init__(self, cache_calls: bool = True):
        super().__init__(cache_calls=cache_calls)
        self.counter = 0
        self.has_finished = False

    def run(self):
        self.counter = self.counter + 1
        if self.cache_calls or self.counter >= 3:
            self.has_finished = True


class SimpleFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_a = Work(cache_calls=True)
        self.work_b = Work(cache_calls=False)

    def run(self):
        if self.work_a.has_finished and self.work_b.has_finished:
            self.stop()
        self.work_a.run()
        self.work_b.run()


def test_simple_app(tmpdir):
    comp = SimpleFlow()
    app = LightningApp(comp, log_level="debug")
    assert app.root == comp
    expected = {
        "app_state": mock.ANY,
        "vars": {"_layout": mock.ANY, "_paths": {}},
        "calls": {},
        "flows": {},
        "structures": {},
        "works": {
            "work_b": {
                "vars": {
                    "has_finished": False,
                    "counter": 0,
                    "_cloud_compute": mock.ANY,
                    "_host": mock.ANY,
                    "_url": "",
                    "_future_url": "",
                    "_internal_ip": "",
                    "_public_ip": "",
                    "_paths": {},
                    "_port": None,
                    "_restarting": False,
                    "_display_name": "",
                },
                "calls": {"latest_call_hash": None},
                "changes": {},
            },
            "work_a": {
                "vars": {
                    "has_finished": False,
                    "counter": 0,
                    "_cloud_compute": mock.ANY,
                    "_host": mock.ANY,
                    "_url": "",
                    "_future_url": "",
                    "_internal_ip": "",
                    "_public_ip": "",
                    "_paths": {},
                    "_port": None,
                    "_restarting": False,
                    "_display_name": "",
                },
                "calls": {"latest_call_hash": None},
                "changes": {},
            },
        },
        "changes": {},
    }
    assert app.state == expected
    MultiProcessRuntime(app, start_server=False).dispatch()

    assert comp.work_a.has_finished
    assert comp.work_b.has_finished
    # possible the `work_a` takes for ever to
    # start and `work_b` has already completed multiple iterations.
    assert comp.work_a.counter == 1
    assert comp.work_b.counter >= 3


class WorkCounter(LightningWork):
    def __init__(self):
        super().__init__()
        self.c = 0

    def run(self):
        self.c = 1


class E(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w_e = WorkCounter()

    def run(self):
        self.w_e.run()


class D(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w_d = WorkCounter()
        self.e = E()

    def run(self):
        self.w_d.run()
        self.e.run()


class C(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w_c = WorkCounter()
        self.d = D()

    def run(self):
        self.w_c.run()
        self.d.run()


class B(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w_b = WorkCounter()
        self.c = C()

    def run(self):
        self.w_b.run()
        self.c.run()


class A(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w_a = WorkCounter()
        self.b = B()

    def run(self):
        self.w_a.run()
        self.b.run()
        if self.b.c.d.e.w_e.c == 1:
            self.stop()


def test_nested_component_names():
    root = A()
    assert root.name == "root"
    assert root.w_a.name == "root.w_a"
    assert root.b.name == "root.b"
    assert root.b.w_b.name == "root.b.w_b"
    assert root.b.c.name == "root.b.c"
    assert root.b.c.w_c.name == "root.b.c.w_c"
    assert root.b.c.d.name == "root.b.c.d"
    assert root.b.c.d.e.name == "root.b.c.d.e"
    assert root.b.c.d.e.w_e.name == "root.b.c.d.e.w_e"
    assert root.b.c.d.w_d.name == "root.b.c.d.w_d"


def test_get_component_by_name():
    app = LightningApp(A())
    assert app.root in app.flows
    assert app.get_component_by_name("root") is app.root
    assert app.get_component_by_name("root.b") is app.root.b
    assert app.get_component_by_name("root.w_a") is app.root.w_a
    assert app.get_component_by_name("root.b.w_b") is app.root.b.w_b
    assert app.get_component_by_name("root.b.c.d.e") is app.root.b.c.d.e


def test_get_component_by_name_raises():
    app = LightningApp(A())

    for name in ("", "ro", "roott"):
        with pytest.raises(ValueError, match=f"Invalid component name {name}."):
            app.get_component_by_name(name)

    with pytest.raises(AttributeError, match="Component 'root' has no child component with name ''"):
        app.get_component_by_name("root.")

    with pytest.raises(AttributeError, match="Component 'root' has no child component with name 'x'"):
        app.get_component_by_name("root.x")

    with pytest.raises(AttributeError, match="Component 'root.b' has no child component with name 'x'"):
        app.get_component_by_name("root.b.x")

    with pytest.raises(AttributeError, match="Component 'root.b.w_b' has no child component with name 'c'"):
        app.get_component_by_name("root.b.w_b.c")


def test_nested_component():
    app = LightningApp(A(), log_level="debug")
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.root.w_a.c == 1
    assert app.root.b.w_b.c == 1
    assert app.root.b.c.w_c.c == 1
    assert app.root.b.c.d.w_d.c == 1
    assert app.root.b.c.d.e.w_e.c == 1


class WorkCCC(LightningWork):
    def run(self):
        pass


class CC(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_cc = WorkCCC()

    def run(self):
        pass


class BB(LightningFlow):
    def __init__(self):
        super().__init__()
        self.c1 = CC()
        self.c2 = CC()

    def run(self):
        pass


class AA(LightningFlow):
    def __init__(self):
        super().__init__()
        self.b = BB()

    def run(self):
        pass


def test_component_affiliation():
    app = LightningApp(AA())
    a_affiliation = affiliation(app.root)
    assert a_affiliation == ()
    b_affiliation = affiliation(app.root.b)
    assert b_affiliation == ("b",)
    c1_affiliation = affiliation(app.root.b.c1)
    assert c1_affiliation == ("b", "c1")
    c2_affiliation = affiliation(app.root.b.c2)
    assert c2_affiliation == ("b", "c2")
    work_cc_affiliation = affiliation(app.root.b.c2.work_cc)
    assert work_cc_affiliation == ("b", "c2", "work_cc")


class Work4(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.var_a = 0
        self.has_finished = False

    def run(self):
        self.var_a = 1
        sleep(2)
        # This would never been reached as the app would exit before
        self.has_finished = True


class A4(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = Work4()

    def run(self):
        self.work.run()
        if self.work.var_a == 1:
            self.stop()


@pytest.mark.parametrize("runtime_cls", [MultiProcessRuntime])
def test_setattr_multiprocessing(runtime_cls, tmpdir):
    app = LightningApp(A4())
    runtime_cls(app, start_server=False).dispatch()
    assert app.root.work.var_a == 1
    assert not app.root.work.has_finished


class CounterFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        self.counter += 1


class SimpleApp2(LightningApp):
    def run_once(self):
        if self.root.counter == 5:
            self.stage = AppStage.RESTARTING
        return super().run_once()

    def _apply_restarting(self):
        super()._apply_restarting()
        assert self.stage == AppStage.BLOCKING
        return True


def test_app_restarting_move_to_blocking(tmpdir):
    """Validates sending restarting move the app to blocking again."""
    app = SimpleApp2(CounterFlow(), log_level="debug")
    MultiProcessRuntime(app, start_server=False).dispatch()


class FlowWithFrontend(LightningFlow):
    def run(self):
        pass

    def configure_layout(self):
        return StreamlitFrontend(render_fn=lambda _: None)


class AppWithFrontend(LightningApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_once_call_count = 0

    def run_once(self):
        # by the time run_once gets called the first time, the target_url for the frontend should be set
        # and be present in both the LightningApp.state and the LightningApp._original_state
        assert self.state["vars"]["_layout"]["target"].startswith("http://localhost")
        assert self._original_state["vars"]["_layout"]["target"].startswith("http://localhost")
        assert self.run_once_call_count or self.state == self._original_state

        self.run_once_call_count += 1
        if self.run_once_call_count == 3:
            return True, 0.0
        return super().run_once()


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="requires streamlit")
@mock.patch("lightning.app.frontend.stream_lit.StreamlitFrontend.start_server")
@mock.patch("lightning.app.frontend.stream_lit.StreamlitFrontend.stop_server")
def test_app_starts_with_complete_state_copy(_, __):
    """Test that the LightningApp captures the initial state in a separate copy when _run() gets called."""
    app = AppWithFrontend(FlowWithFrontend(), log_level="debug")
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.run_once_call_count == 3


class EmptyFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        pass


@pytest.mark.parametrize(
    ("queue_type_cls", "default_timeout"),
    [
        (MultiProcessQueue, STATE_UPDATE_TIMEOUT),
        pytest.param(
            RedisQueue,
            REDIS_QUEUES_READ_DEFAULT_TIMEOUT,
            marks=pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running"),
        ),
    ],
)
@pytest.mark.parametrize(
    ("sleep_time", "expect"),
    [
        (0, 9),
        pytest.param(9, 10.0, marks=pytest.mark.xfail(strict=False, reason="failing...")),  # fixme
    ],
)
@pytest.mark.flaky(reruns=5)
def test_lightning_app_aggregation_speed(default_timeout, queue_type_cls: BaseQueue, sleep_time, expect):
    """This test validates the `_collect_deltas_from_ui_and_work_queues` can aggregate multiple delta together in a
    time window."""

    class SlowQueue(queue_type_cls):
        def batch_get(self, timeout, count):
            out = super().get(timeout)
            sleep(sleep_time)
            return [out]

    app = LightningApp(EmptyFlow())

    app.delta_queue = SlowQueue("api_delta_queue", default_timeout)
    if queue_type_cls is RedisQueue:
        app.delta_queue.clear()

    def make_delta(i):
        return _DeltaRequest(Delta({"values_changed": {"root['vars']['counter']": {"new_value": i}}}))

    # flowed the queue with mocked delta
    for i in range(expect + 10):
        app.delta_queue.put(make_delta(i))

    # Wait for a bit because multiprocessing.Queue doesn't run in the same thread and takes some time for writes
    sleep(0.001)

    delta = app._collect_deltas_from_ui_and_work_queues()[-1]
    generated = delta.to_dict()["values_changed"]["root['vars']['counter']"]["new_value"]
    if sleep_time:
        assert generated == expect, (generated, expect)
    else:
        # validate the flow should have aggregated at least expect.
        assert generated > expect


def test_lightning_app_aggregation_empty():
    """Verify the while loop exits before `state_accumulate_wait` is reached if no deltas are found."""

    class SlowQueue(MultiProcessQueue):
        def get(self, timeout):
            return super().get(timeout)

    app = LightningApp(EmptyFlow())
    app.delta_queue = SlowQueue("api_delta_queue", 0)
    t0 = time()
    assert app._collect_deltas_from_ui_and_work_queues() == []
    delta = time() - t0
    assert delta < app.state_accumulate_wait + 0.01, delta


class SimpleFlow2(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        if self.counter < 2:
            self.counter += 1


def test_maybe_apply_changes_from_flow():
    """This test validates the app `_updated` is set to True only if the state was changed in the flow."""
    app = LightningApp(SimpleFlow2())
    app.delta_queue = MultiProcessQueue("a", 0)
    assert app._has_updated
    app.maybe_apply_changes()
    app.root.run()
    app.maybe_apply_changes()
    assert app._has_updated
    app._has_updated = False
    app.root.run()
    app.maybe_apply_changes()
    assert app._has_updated
    app._has_updated = False
    app.root.run()
    app.maybe_apply_changes()
    assert not app._has_updated


class SimpleWork(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False, parallel=True)
        self.counter = 0

    def run(self):
        self.counter += 1


class FlowA(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_a = SimpleWork()
        self.work_b = SimpleWork()

    def run(self):
        if self.work_a.counter == self.work_b.counter == 0:
            self.work_a.run()
            self.work_b.run()


class SuccessException(Exception):
    pass


class CheckpointLightningApp(LightningApp):
    def _dump_checkpoint(self):
        super()._dump_checkpoint()
        raise SuccessException


@pytest.mark.flaky(reruns=3)
def test_snap_shotting():
    with contextlib.suppress(SuccessException):
        app = CheckpointLightningApp(FlowA())
        app.checkpointing = True
        MultiProcessRuntime(app, start_server=False).dispatch()

    checkpoint_dir = os.path.join(_storage_root_dir(), "checkpoints")
    checkpoints = os.listdir(checkpoint_dir)
    assert len(checkpoints) == 1
    with open(os.path.join(checkpoint_dir, checkpoints[0]), "rb") as f:
        state = pickle.load(f)
        assert state["works"]["work_a"]["vars"]["counter"] == 1
        assert state["works"]["work_b"]["vars"]["counter"] == 1


class CounterWork(LightningWork):
    def __init__(self, parallel: bool, cache_calls: bool):
        super().__init__(parallel=parallel, cache_calls=cache_calls)
        self.counter = 0

    def run(self, counter=0):
        self.counter += 1


class WaitForAllFlow(LightningFlow):
    def __init__(self, use_same_args):
        super().__init__()
        counter = 0
        self.use_same_args = use_same_args
        for parallel in [False, True]:
            for cache_calls in [False, True]:
                work = CounterWork(parallel=parallel, cache_calls=cache_calls)
                setattr(self, f"work_{counter}", work)
                counter += 1
        self.c = 0

    def run(self):
        next_c = self.c + 1
        for work in self.experimental_iterate(self.works(), run_once=False):
            if work.num_successes < (next_c):
                if not self.use_same_args:
                    work.run(self.c)
                else:
                    work.run(None)

        expected = 1 if self.use_same_args else next_c

        if not all(w.num_successes == (expected if w.cache_calls else next_c) for w in self.works()):
            return

        self.c += 1
        assert [w.counter for w in self.works()] == [self.c, expected, self.c, expected]
        if self.c > 3:
            self.stop()


# TODO (tchaton) Resolve this test.
@pytest.mark.skipif(_IS_WINDOWS, reason="timeout with system crash")
@pytest.mark.xfail(strict=False, reason="flaky test which never terminates")
@pytest.mark.parametrize("runtime_cls", [MultiProcessRuntime])
@pytest.mark.parametrize("use_same_args", [True])
# todo: removed test_state_wait_for_all_all_works[False-MultiProcessRuntime] as it hangs
def test_state_wait_for_all_all_works(tmpdir, runtime_cls, use_same_args):
    app = LightningApp(WaitForAllFlow(use_same_args))
    runtime_cls(app, start_server=False).dispatch()


class CheckpointCounter(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False)
        self.counter = 0

    def run(self):
        self.counter += 1


class CheckpointFlow(LightningFlow):
    def __init__(self, work: CheckpointCounter, depth=0):
        super().__init__()
        self.depth = depth

        if depth == 0:
            self.counter = 0

        if depth >= 10:
            self.work = work
        else:
            self.flow = CheckpointFlow(work, depth + 1)

    def run(self):
        if self.works()[0].counter == 5:
            self.stop()

        if self.depth >= 10:
            self.work.run()
        else:
            self.flow.run()


@pytest.mark.skipif(True, reason="reloading isn't properly supported")
def test_lightning_app_checkpointing_with_nested_flows():
    work = CheckpointCounter()
    app = LightningApp(CheckpointFlow(work))
    app.checkpointing = True
    MultiProcessRuntime(app, start_server=False).dispatch()

    assert app.root.flow.flow.flow.flow.flow.flow.flow.flow.flow.flow.work.counter == 5

    work = CheckpointCounter()
    app = LightningApp(CheckpointFlow(work))
    assert app.root.flow.flow.flow.flow.flow.flow.flow.flow.flow.flow.work.counter == 0

    app.load_state_dict_from_checkpoint_dir(app.checkpoint_dir)
    # The counter was increment to 6 after the latest checkpoints was created.
    assert app.root.flow.flow.flow.flow.flow.flow.flow.flow.flow.flow.work.counter == 5


@pytest.mark.xfail(strict=False, reason="test is skipped because CI was blocking all the PRs.")
def test_load_state_dict_from_checkpoint_dir(tmpdir):
    work = CheckpointCounter()
    app = LightningApp(CheckpointFlow(work))

    checkpoints = []
    num_checkpoints = 11
    # generate 11 checkpoints.
    for _ in range(num_checkpoints):
        checkpoints.append(app._dump_checkpoint())
        app.root.counter += 1

    app.load_state_dict_from_checkpoint_dir(app.checkpoint_dir)
    assert app.root.counter == (num_checkpoints - 1)

    for version in range(num_checkpoints):
        app.load_state_dict_from_checkpoint_dir(app.checkpoint_dir, version=version)
        assert app.root.counter == version

    with pytest.raises(FileNotFoundError, match="The provided directory"):
        app.load_state_dict_from_checkpoint_dir("./random_folder/")

    with pytest.raises(Exception, match="No checkpoints where found"):
        app.load_state_dict_from_checkpoint_dir(str(os.path.join(_PROJECT_ROOT, "tests/tests_app/")))

    # delete 2 checkpoints
    os.remove(os.path.join(checkpoints[4]))
    os.remove(os.path.join(checkpoints[7]))

    app.load_state_dict_from_checkpoint_dir(app.checkpoint_dir)
    assert app.root.counter == (num_checkpoints - 1)

    app.load_state_dict_from_checkpoint_dir(app.checkpoint_dir, version=5)
    checkpoint_path = app._dump_checkpoint()

    assert os.path.basename(checkpoint_path).startswith("v_11")


class PicklableObject:
    pass


class PickleableReturnWork(LightningWork):
    def __init__(self):
        super().__init__()

    def run(self):
        return PicklableObject()


class PickleableReturnFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = PickleableReturnWork()

    def run(self):
        self.work.run()


def test_pickleable_return_from_work():
    """Test that any object that is pickleable can be returned from the run method in LightningWork."""
    with pytest.raises(SystemExit, match="1"):
        app = LightningApp(PickleableReturnFlow())
        MultiProcessRuntime(app, start_server=False).dispatch()


class WorkDD(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.total = 10
        self.counter = 1

    def run(self):
        should_wait = self.counter == 1
        start_counter = self.total - self.counter
        for _ in range(start_counter):
            if should_wait:
                sleep(0.5)
            self.counter += 1


class FlowCCTolerance(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = WorkDD()

    def run(self):
        self.work.run()
        if self.work.counter == 10:
            self.stop()


class FaultToleranceLightningTestApp(LightningTestApp):
    def on_after_run_once(self):
        if self.root.work.status.reason == WorkStopReasons.SIGTERM_SIGNAL_HANDLER:
            assert self.root.work.counter < 10
            self.restart_work("root.work")
        elif self.root.work.counter == 2:
            self.kill_work("root.work")
            return True, 0.0
        return super().on_after_run_once()


# TODO (tchaton) Resolve this test with Resumable App.
@_RunIf(skip_windows=True)
def test_fault_tolerance_work():
    app = FaultToleranceLightningTestApp(FlowCCTolerance())
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.root.work.counter == 2


class ProtectedAttributesWork(LightningWork):
    def __init__(self):
        super().__init__()
        # a public attribute, this should show up in the state
        self.done = False
        # a protected and a private attribute, these should NOT show up in the state
        self._protected = 1
        self.__private = 2

    def run(self):
        self.done = True
        self._protected = 10
        self.__private = 20


class ProtectedAttributesFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        # a public attribute, this should show up in the state
        self.done = False
        # a protected and a private attribute, these should NOT show up in the state
        self._protected = 1
        self.__private = 2

        self.protected_work = ProtectedAttributesWork()

    def run(self):
        flow_variables = self.state_vars["vars"]
        assert "done" in flow_variables
        assert "_protected" not in flow_variables
        assert "__private" not in flow_variables
        self.done = True

        self.protected_work.run()
        if self.protected_work.done:
            work_variables = self.protected_work.state_vars["vars"]
            assert "done" in work_variables
            assert "_protected" not in work_variables
            assert "__private" not in work_variables

            # TODO: getattr and setattr access outside the Work should raise an error in the future
            _ = self.protected_work._protected
            self.protected_work._protected = 1

        if self.done and self.protected_work.done:
            self.stop()


def test_protected_attributes_not_in_state():
    flow = ProtectedAttributesFlow()
    MultiProcessRuntime(LightningApp(flow), start_server=False).dispatch()


class WorkExit(LightningWork):
    def __init__(self):
        super().__init__(raise_exception=False)
        self.counter = 0

    def run(self):
        self.counter += 1
        raise Exception("Hello")


class FlowExit(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = WorkExit()

    def run(self):
        if self.work.counter == 1:
            self.stop()
        self.work.run()


def test_lightning_app_exit():
    app = LightningApp(FlowExit())
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.root.work.status.stage == WorkStageStatus.STOPPED


class CounterWork2(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.counter = 0

    def run(self):
        self.counter += 1


class FlowStop(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = CounterWork2()

    def run(self):
        if self.w.status.stage == WorkStageStatus.STOPPED:
            self.stop()
        if self.w.counter == 1:
            self.w.stop()
        self.w.run()


@_RunIf(skip_windows=True)
def test_lightning_stop():
    app = LightningApp(FlowStop())
    MultiProcessRuntime(app, start_server=False).dispatch()


class SleepyFlow(LightningFlow):
    def __init__(self, sleep_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.sleep_interval = sleep_interval

    def run(self):
        if self.counter == 2 * FLOW_DURATION_SAMPLES:
            self.stop()
        sleep(self.sleep_interval)
        self.counter += 1


class SleepyWork(LightningWork):
    def __init__(self, sleep_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_interval = sleep_interval

    def run(self):
        sleep(self.sleep_interval)


class SleepyFlowWithWork(LightningFlow):
    def __init__(self, sleep_interval, work_sleep_interval, parallel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.sleep_interval = sleep_interval
        self.work = SleepyWork(work_sleep_interval, parallel=parallel)

    def run(self):
        if self.counter == 2 * FLOW_DURATION_SAMPLES:
            self.stop()
        self.work.run()
        sleep(self.sleep_interval)
        self.counter += 1


def test_slow_flow():
    app0 = LightningApp(SleepyFlow(sleep_interval=0.5 * FLOW_DURATION_THRESHOLD))

    MultiProcessRuntime(app0, start_server=False).dispatch()

    app1 = LightningApp(SleepyFlow(sleep_interval=2 * FLOW_DURATION_THRESHOLD))

    with pytest.warns(LightningFlowWarning):
        MultiProcessRuntime(app1, start_server=False).dispatch()

    app0 = LightningApp(
        SleepyFlowWithWork(
            sleep_interval=0.5 * FLOW_DURATION_THRESHOLD,
            work_sleep_interval=2 * FLOW_DURATION_THRESHOLD,
            parallel=False,
        )
    )

    MultiProcessRuntime(app0, start_server=False).dispatch()

    app1 = LightningApp(
        SleepyFlowWithWork(
            sleep_interval=0.5 * FLOW_DURATION_THRESHOLD, work_sleep_interval=2 * FLOW_DURATION_THRESHOLD, parallel=True
        )
    )

    MultiProcessRuntime(app1, start_server=False).dispatch()


class SizeWork(LightningWork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0

    def run(self, signal: int):
        self.counter += 1
        assert len(self._calls) == 2


class SizeFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work0 = SizeWork(parallel=True, cache_calls=True)
        self._state_sizes = {}

    def run(self):
        for idx in range(self.work0.counter + 2):
            self.work0.run(idx)

        self._state_sizes[self.work0.counter] = asizeof.asizeof(self.state)

        if self.work0.counter >= 20:
            self.stop()


def test_state_size_constant_growth():
    app = LightningApp(SizeFlow())
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.root._state_sizes[0] <= 8380
    assert app.root._state_sizes[20] <= 26999


class FlowUpdated(LightningFlow):
    def run(self):
        logger.info("Hello World")


class NonUpdatedLightningTestApp(LightningTestApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def on_after_run_once(self):
        self.counter += 1
        if not self._has_updated and self.counter > 2:
            return True
        return super().on_after_run_once()


def test_non_updated_flow(caplog):
    """Validate that the app can run 3 times and calls the flow only once."""
    app = NonUpdatedLightningTestApp(FlowUpdated())
    runtime = MultiProcessRuntime(app, start_server=False)
    with caplog.at_level(logging.INFO):
        runtime.dispatch()
    assert caplog.messages == [
        "Hello World",
        "Your Lightning App is being stopped. This won't take long.",
        "Your Lightning App has been stopped successfully!",
    ]
    assert app.counter == 3


def test_debug_mode_logging():
    """This test validates the DEBUG messages are collected when activated by the LightningApp(debug=True) and cleanup
    once finished."""

    from lightning.app.core.app import _console

    app = LightningApp(A4(), log_level="debug")
    assert _console.level == logging.DEBUG
    assert os.getenv("LIGHTNING_DEBUG") == "2"

    MultiProcessRuntime(app, start_server=False).dispatch()

    assert os.getenv("LIGHTNING_DEBUG") is None
    assert _console.level == logging.INFO

    app = LightningApp(A4())
    assert _console.level == logging.INFO
    MultiProcessRuntime(app, start_server=False).dispatch()


class WorkPath(LightningWork):
    def __init__(self):
        super().__init__()
        self.path = None

    def run(self):
        self.path = Path(__file__)


class FlowPath(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w = WorkPath()

    def run(self):
        self.w.run()


class TestLightningHasUpdatedApp(LightningApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def run_once(self):
        res = super().run_once()

        if self.root.w.has_succeeded:
            self.counter += 1

        # TODO: Resolve bug where it should work with self.counter == 2
        if self.counter > 5:
            assert not self._has_updated
            return True
        return res


@pytest.mark.flaky(reruns=3)
def test_lightning_app_has_updated():
    app = TestLightningHasUpdatedApp(FlowPath())
    MultiProcessRuntime(app, start_server=False).dispatch()


class WorkCC(LightningWork):
    def run(self):
        pass


class FlowCC(LightningFlow):
    def __init__(self):
        super().__init__()
        self.cloud_compute = CloudCompute(name="gpu", _internal_id="a")
        self.work_a = WorkCC(cloud_compute=self.cloud_compute)
        self.work_b = WorkCC(cloud_compute=self.cloud_compute)
        self.work_c = WorkCC()
        assert self.work_a.cloud_compute._internal_id == self.work_b.cloud_compute._internal_id

    def run(self):
        self.work_d = WorkCC()


class FlowWrapper(LightningFlow):
    def __init__(self, flow):
        super().__init__()
        self.w = flow


def test_cloud_compute_binding():
    cloud_compute.ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER = True

    assert {} == cloud_compute._CLOUD_COMPUTE_STORE
    flow = FlowCC()
    assert len(cloud_compute._CLOUD_COMPUTE_STORE) == 2
    assert cloud_compute._CLOUD_COMPUTE_STORE["default"].component_names == ["root.work_c"]
    assert cloud_compute._CLOUD_COMPUTE_STORE["a"].component_names == ["root.work_a", "root.work_b"]

    wrapper = FlowWrapper(flow)
    assert cloud_compute._CLOUD_COMPUTE_STORE["default"].component_names == ["root.w.work_c"]
    assert cloud_compute._CLOUD_COMPUTE_STORE["a"].component_names == ["root.w.work_a", "root.w.work_b"]

    _ = FlowWrapper(wrapper)
    assert cloud_compute._CLOUD_COMPUTE_STORE["default"].component_names == ["root.w.w.work_c"]
    assert cloud_compute._CLOUD_COMPUTE_STORE["a"].component_names == ["root.w.w.work_a", "root.w.w.work_b"]

    assert flow.state["vars"]["cloud_compute"]["type"] == "__cloud_compute__"
    assert flow.work_a.state["vars"]["_cloud_compute"]["type"] == "__cloud_compute__"
    assert flow.work_b.state["vars"]["_cloud_compute"]["type"] == "__cloud_compute__"
    assert flow.work_c.state["vars"]["_cloud_compute"]["type"] == "__cloud_compute__"
    work_a_id = flow.work_a.state["vars"]["_cloud_compute"]["_internal_id"]
    work_b_id = flow.work_b.state["vars"]["_cloud_compute"]["_internal_id"]
    work_c_id = flow.work_c.state["vars"]["_cloud_compute"]["_internal_id"]
    assert work_a_id == work_b_id
    assert work_a_id != work_c_id
    assert work_c_id == "default"

    flow.work_a.cloud_compute = CloudCompute(name="something_else")
    assert cloud_compute._CLOUD_COMPUTE_STORE["a"].component_names == ["root.w.w.work_b"]

    flow.set_state(flow.state)
    assert isinstance(flow.cloud_compute, CloudCompute)
    assert isinstance(flow.work_a.cloud_compute, CloudCompute)
    assert isinstance(flow.work_c.cloud_compute, CloudCompute)

    cloud_compute.ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER = False

    with pytest.raises(Exception, match="A Cloud Compute can be assigned only to a single Work"):
        FlowCC()


class FlowValue(LightningFlow):
    def __init__(self):
        super().__init__()
        self._value = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def run(self):
        self.value = True


def test_lightning_flow_properties():
    """Validates setting properties to the LightningFlow properly calls property.fset."""
    flow = FlowValue()
    assert flow._value is None
    flow.run()
    assert flow._value is True


class SimpleWork2(LightningWork):
    def run(self):
        pass


def test_lightning_work_stopped():
    app = LightningApp(SimpleWork2())
    MultiProcessRuntime(app, start_server=False).dispatch()


class FailedWork(LightningWork):
    def run(self):
        raise Exception


class CheckErrorQueueLightningApp(LightningApp):
    def check_error_queue(self):
        super().check_error_queue()


def test_error_queue_check(monkeypatch):
    import sys

    from lightning.app.core import app as app_module

    sys_mock = mock.MagicMock()
    monkeypatch.setattr(app_module, "CHECK_ERROR_QUEUE_INTERVAL", 0)
    monkeypatch.setattr(sys, "exit", sys_mock)
    app = LightningApp(FailedWork())
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.stage == AppStage.FAILED
    assert app._last_check_error_queue != 0.0
