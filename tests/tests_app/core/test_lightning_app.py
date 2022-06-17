import os
import pickle
from time import sleep
from unittest import mock
from unittest.mock import ANY

import pytest
from deepdiff import Delta

from lightning_app import LightningApp, LightningFlow, LightningWork  # F401
from lightning_app.core.constants import (
    FLOW_DURATION_SAMPLES,
    FLOW_DURATION_THRESHOLD,
    REDIS_QUEUES_READ_DEFAULT_TIMEOUT,
    STATE_UPDATE_TIMEOUT,
)
from lightning_app.core.queues import BaseQueue, MultiProcessQueue, RedisQueue, SingleProcessQueue
from lightning_app.frontend import StreamlitFrontend
from lightning_app.runners import MultiProcessRuntime, SingleProcessRuntime
from lightning_app.storage.path import storage_root_dir
from lightning_app.testing.helpers import RunIf
from lightning_app.testing.testing import LightningTestApp
from lightning_app.utilities.app_helpers import affiliation
from lightning_app.utilities.enum import AppStage, WorkStageStatus, WorkStopReasons
from lightning_app.utilities.redis import check_if_redis_running
from lightning_app.utilities.warnings import LightningFlowWarning


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
        self.counter += 1
        if self.cache_calls:
            self.has_finished = True
        elif self.counter >= 3:
            self.has_finished = True


class SimpleFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_a = Work(cache_calls=True)
        self.work_b = Work(cache_calls=False)

    def run(self):
        self.work_a.run()
        self.work_b.run()
        if self.work_a.has_finished and self.work_b.has_finished:
            self._exit()


@pytest.mark.skip
@pytest.mark.parametrize("component_cls", [SimpleFlow])
@pytest.mark.parametrize("runtime_cls", [SingleProcessRuntime])
def test_simple_app(component_cls, runtime_cls, tmpdir):
    comp = component_cls()
    app = LightningApp(comp, debug=True)
    assert app.root == comp
    expected = {
        "app_state": ANY,
        "vars": {"_layout": ANY, "_paths": {}},
        "calls": {},
        "flows": {},
        "works": {
            "work_b": {
                "vars": {"has_finished": False, "counter": 0, "_urls": {}, "_paths": {}},
                "calls": {},
                "changes": {},
            },
            "work_a": {
                "vars": {"has_finished": False, "counter": 0, "_urls": {}, "_paths": {}},
                "calls": {},
                "changes": {},
            },
        },
        "changes": {},
    }
    assert app.state == expected
    runtime_cls(app, start_server=False).dispatch()

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
            self._exit()


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


@pytest.mark.parametrize("runtime_cls", [SingleProcessRuntime, MultiProcessRuntime])
def test_nested_component(runtime_cls):
    app = LightningApp(A(), debug=True)
    runtime_cls(app, start_server=False).dispatch()
    assert app.root.w_a.c == 1
    assert app.root.b.w_b.c == 1
    assert app.root.b.c.w_c.c == 1
    assert app.root.b.c.d.w_d.c == 1
    assert app.root.b.c.d.e.w_e.c == 1


class WorkCC(LightningWork):
    def run(self):
        pass


class CC(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work_cc = WorkCC()

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
            self._exit()


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


@pytest.mark.parametrize("runtime_cls", [SingleProcessRuntime, MultiProcessRuntime])
def test_app_restarting_move_to_blocking(runtime_cls, tmpdir):
    """Validates sending restarting move the app to blocking again."""
    app = SimpleApp2(CounterFlow(), debug=True)
    runtime_cls(app, start_server=False).dispatch()


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


@mock.patch("lightning_app.frontend.stream_lit.StreamlitFrontend.start_server")
@mock.patch("lightning_app.frontend.stream_lit.StreamlitFrontend.stop_server")
def test_app_starts_with_complete_state_copy(_, __):
    """Test that the LightningApp captures the initial state in a separate copy when _run() gets called."""
    app = AppWithFrontend(FlowWithFrontend(), debug=True)
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.run_once_call_count == 3


class EmptyFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        pass


@pytest.mark.parametrize(
    "queue_type_cls, default_timeout",
    [
        (SingleProcessQueue, STATE_UPDATE_TIMEOUT),
        (MultiProcessQueue, STATE_UPDATE_TIMEOUT),
        pytest.param(
            RedisQueue,
            REDIS_QUEUES_READ_DEFAULT_TIMEOUT,
            marks=pytest.mark.skipif(not check_if_redis_running(), reason="Redis is not running"),
        ),
    ],
)
@pytest.mark.parametrize(
    "sleep_time, expect",
    [
        (1, 0),
        (0, 100),
    ],
)
def test_lightning_app_aggregation_speed(default_timeout, queue_type_cls: BaseQueue, sleep_time, expect):

    """This test validates the `_collect_deltas_from_ui_and_work_queues` can aggregate multiple delta together in a time
    window."""

    class SlowQueue(queue_type_cls):
        def get(self, timeout):
            out = super().get(timeout)
            sleep(sleep_time)
            return out

    app = LightningApp(EmptyFlow())

    app.api_delta_queue = SlowQueue("api_delta_queue", default_timeout)
    if queue_type_cls is RedisQueue:
        app.api_delta_queue.clear()

    def make_delta(i):
        return Delta({"values_changed": {"root['vars']['counter']": {"new_value": i}}})

    # flowed the queue with mocked delta
    for i in range(expect + 10):
        app.api_delta_queue.put(make_delta(i))

    # Wait for a bit because multiprocessing.Queue doesn't run in the same thread and takes some time for writes
    sleep(0.001)

    delta = app._collect_deltas_from_ui_and_work_queues()[-1]
    generated = delta.to_dict()["values_changed"]["root['vars']['counter']"]["new_value"]
    if sleep_time:
        assert generated == expect
    else:
        # validate the flow should have aggregated at least expect.
        assert generated > expect


class SimpleFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        self.counter = 1


def test_maybe_apply_changes_from_flow():
    """This test validates the app `_updated` is set to True only if the state was changed in the flow."""

    app = LightningApp(SimpleFlow())
    assert not app._has_updated
    app.maybe_apply_changes()
    app.root.run()
    app.maybe_apply_changes()
    assert app._has_updated
    app._has_updated = False
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


@pytest.mark.parametrize("runtime_cls", [MultiProcessRuntime])
def test_snapshotting(runtime_cls, tmpdir):
    try:
        app = CheckpointLightningApp(FlowA())
        app.checkpointing = True
        runtime_cls(app, start_server=False).dispatch()
    except SuccessException:
        pass
    checkpoint_dir = os.path.join(storage_root_dir(), "checkpoints")
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

        if not all([w.num_successes == (expected if w.cache_calls else next_c) for w in self.works()]):
            return

        self.c += 1
        assert [w.counter for w in self.works()] == [self.c, expected, self.c, expected]
        if self.c > 3:
            self._exit()


# TODO (tchaton) Resolve this test.
@pytest.mark.skipif(True, reason="flaky test which never terminates")
@pytest.mark.parametrize("runtime_cls", [MultiProcessRuntime])
@pytest.mark.parametrize("use_same_args", [False, True])
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
    def __init__(self, work: LightningWork, depth=0):
        super().__init__()
        self.depth = depth
        if depth == 0:
            self.counter = 0

        if depth >= 10:
            self.work = work
        else:
            self.flow = CheckpointFlow(work, depth + 1)

    def run(self):
        if hasattr(self, "counter"):
            self.counter += 1
            if self.counter > 5:
                self._exit()
        if self.depth >= 10:
            self.work.run()
        else:
            self.flow.run()


def test_lightning_app_checkpointing_with_nested_flows():
    work = CheckpointCounter()
    app = LightningApp(CheckpointFlow(work))
    app.checkpointing = True
    SingleProcessRuntime(app, start_server=False).dispatch()

    assert app.root.counter == 6
    assert app.root.flow.flow.flow.flow.flow.flow.flow.flow.flow.flow.work.counter == 5

    work = CheckpointCounter()
    app = LightningApp(CheckpointFlow(work))
    assert app.root.counter == 0
    assert app.root.flow.flow.flow.flow.flow.flow.flow.flow.flow.flow.work.counter == 0

    app.load_state_dict_from_checkpoint_dir(app.checkpoint_dir)
    # The counter was increment to 6 after the latest checkpoints was created.
    assert app.root.counter == 5
    assert app.root.flow.flow.flow.flow.flow.flow.flow.flow.flow.flow.work.counter == 5


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
        app.load_state_dict_from_checkpoint_dir("./tests")

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


class FlowCC(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = WorkDD()

    def run(self):
        self.work.run()
        if self.work.counter == 10:
            self._exit()


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
@RunIf(skip_windows=True)
def test_fault_tolerance_work():
    app = FaultToleranceLightningTestApp(FlowCC())
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
            self._exit()


def test_protected_attributes_not_in_state():
    flow = ProtectedAttributesFlow()
    MultiProcessRuntime(LightningApp(flow)).dispatch()


class WorkExit(LightningWork):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class FlowExit(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = WorkExit()

    def run(self):
        self.work.run()
        self._exit()


def test_lightning_app_exit():
    app = LightningApp(FlowExit())
    MultiProcessRuntime(app).dispatch()
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
            self._exit()
        if self.w.counter == 1:
            self.w.stop()
        self.w.run()


@RunIf(skip_windows=True)
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
            self._exit()
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
            self._exit()
        self.work.run()
        sleep(self.sleep_interval)
        self.counter += 1


def test_slow_flow():
    app0 = LightningApp(SleepyFlow(sleep_interval=0.5 * FLOW_DURATION_THRESHOLD))

    MultiProcessRuntime(app0).dispatch()

    app1 = LightningApp(SleepyFlow(sleep_interval=2 * FLOW_DURATION_THRESHOLD))

    with pytest.warns(LightningFlowWarning):
        MultiProcessRuntime(app1).dispatch()

    app0 = LightningApp(
        SleepyFlowWithWork(
            sleep_interval=0.5 * FLOW_DURATION_THRESHOLD,
            work_sleep_interval=2 * FLOW_DURATION_THRESHOLD,
            parallel=False,
        )
    )

    MultiProcessRuntime(app0).dispatch()

    app1 = LightningApp(
        SleepyFlowWithWork(
            sleep_interval=0.5 * FLOW_DURATION_THRESHOLD, work_sleep_interval=2 * FLOW_DURATION_THRESHOLD, parallel=True
        )
    )

    MultiProcessRuntime(app1).dispatch()
