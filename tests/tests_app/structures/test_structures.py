import os
from copy import deepcopy

import pytest
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.runners import MultiProcessRuntime
from lightning.app.storage.payload import Payload
from lightning.app.structures import Dict, List
from lightning.app.testing.helpers import EmptyFlow
from lightning.app.utilities.enum import CacheCallsKeys, WorkStageStatus
from lightning.app.utilities.imports import _IS_WINDOWS


def test_dict():
    class WorkA(LightningWork):
        def __init__(self):
            super().__init__(port=1)
            self.c = 0

        def run(self):
            pass

    class A(LightningFlow):
        def __init__(self):
            super().__init__()
            self.dict = Dict(**{"work_a": WorkA(), "work_b": WorkA(), "work_c": WorkA(), "work_d": WorkA()})

        def run(self):
            pass

    flow = A()

    # TODO: these assertions are wrong, the works are getting added under "flows" instead of "works"
    # state
    assert len(flow.state["structures"]["dict"]["works"]) == len(flow.dict) == 4
    assert list(flow.state["structures"]["dict"]["works"].keys()) == ["work_a", "work_b", "work_c", "work_d"]
    assert all(
        flow.state["structures"]["dict"]["works"][f"work_{k}"]["vars"]
        == {
            "c": 0,
            "_url": "",
            "_future_url": "",
            "_port": 1,
            "_host": "127.0.0.1",
            "_paths": {},
            "_restarting": False,
            "_display_name": "",
            "_internal_ip": "",
            "_public_ip": "",
            "_cloud_compute": {
                "type": "__cloud_compute__",
                "name": "cpu-small",
                "disk_size": 0,
                "idle_timeout": None,
                "mounts": None,
                "shm_size": 0,
                "_internal_id": "default",
                "interruptible": False,
                "colocation_group_id": None,
            },
        }
        for k in ("a", "b", "c", "d")
    )
    assert all(
        flow.state["structures"]["dict"]["works"][f"work_{k}"]["calls"] == {CacheCallsKeys.LATEST_CALL_HASH: None}
        for k in ("a", "b", "c", "d")
    )
    assert all(flow.state["structures"]["dict"]["works"][f"work_{k}"]["changes"] == {} for k in ("a", "b", "c", "d"))

    # state_vars
    assert len(flow.state_vars["structures"]["dict"]["works"]) == len(flow.dict) == 4
    assert list(flow.state_vars["structures"]["dict"]["works"].keys()) == ["work_a", "work_b", "work_c", "work_d"]
    assert all(
        flow.state_vars["structures"]["dict"]["works"][f"work_{k}"]["vars"]
        == {
            "c": 0,
            "_url": "",
            "_future_url": "",
            "_port": 1,
            "_host": "127.0.0.1",
            "_paths": {},
            "_restarting": False,
            "_display_name": "",
            "_internal_ip": "",
            "_public_ip": "",
            "_cloud_compute": {
                "type": "__cloud_compute__",
                "name": "cpu-small",
                "disk_size": 0,
                "idle_timeout": None,
                "mounts": None,
                "shm_size": 0,
                "_internal_id": "default",
                "interruptible": False,
                "colocation_group_id": None,
            },
        }
        for k in ("a", "b", "c", "d")
    )

    # state_with_changes
    assert len(flow.state_with_changes["structures"]["dict"]["works"]) == len(flow.dict) == 4
    assert list(flow.state_with_changes["structures"]["dict"]["works"].keys()) == [
        "work_a",
        "work_b",
        "work_c",
        "work_d",
    ]
    assert all(
        flow.state_with_changes["structures"]["dict"]["works"][f"work_{k}"]["vars"]
        == {
            "c": 0,
            "_url": "",
            "_future_url": "",
            "_port": 1,
            "_host": "127.0.0.1",
            "_paths": {},
            "_restarting": False,
            "_display_name": "",
            "_internal_ip": "",
            "_public_ip": "",
            "_cloud_compute": {
                "type": "__cloud_compute__",
                "name": "cpu-small",
                "disk_size": 0,
                "idle_timeout": None,
                "mounts": None,
                "shm_size": 0,
                "_internal_id": "default",
                "interruptible": False,
                "colocation_group_id": None,
            },
        }
        for k in ("a", "b", "c", "d")
    )
    assert all(
        flow.state_with_changes["structures"]["dict"]["works"][f"work_{k}"]["calls"]
        == {CacheCallsKeys.LATEST_CALL_HASH: None}
        for k in ("a", "b", "c", "d")
    )
    assert all(
        flow.state_with_changes["structures"]["dict"]["works"][f"work_{k}"]["changes"] == {}
        for k in ("a", "b", "c", "d")
    )

    # set_state
    state = deepcopy(flow.state)
    state["structures"]["dict"]["works"]["work_b"]["vars"]["c"] = 1
    flow.set_state(state)
    assert flow.dict["work_b"].c == 1


def test_dict_name():
    d = Dict(a=EmptyFlow(), b=EmptyFlow())
    assert d.name == "root"
    assert d["a"].name == "root.a"
    assert d["b"].name == "root.b"

    class RootFlow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.dict = Dict(x=EmptyFlow(), y=EmptyFlow())

        def run(self):
            pass

    root = RootFlow()
    assert root.name == "root"
    assert root.dict.name == "root.dict"
    assert root.dict["x"].name == "root.dict.x"
    assert root.dict["y"].name == "root.dict.y"


def test_list():
    class WorkA(LightningWork):
        def __init__(self):
            super().__init__(port=1)
            self.c = 0

        def run(self):
            pass

    class A(LightningFlow):
        def __init__(self):
            super().__init__()
            self.list = List(WorkA(), WorkA(), WorkA(), WorkA())

        def run(self):
            pass

    flow = A()

    # TODO: these assertions are wrong, the works are getting added under "flows" instead of "works"
    # state
    assert len(flow.state["structures"]["list"]["works"]) == len(flow.list) == 4
    assert list(flow.state["structures"]["list"]["works"].keys()) == ["0", "1", "2", "3"]
    assert all(
        flow.state["structures"]["list"]["works"][str(i)]["vars"]
        == {
            "c": 0,
            "_url": "",
            "_future_url": "",
            "_port": 1,
            "_host": "127.0.0.1",
            "_paths": {},
            "_restarting": False,
            "_internal_ip": "",
            "_public_ip": "",
            "_display_name": "",
            "_cloud_compute": {
                "type": "__cloud_compute__",
                "name": "cpu-small",
                "disk_size": 0,
                "idle_timeout": None,
                "mounts": None,
                "shm_size": 0,
                "_internal_id": "default",
                "interruptible": False,
                "colocation_group_id": None,
            },
        }
        for i in range(4)
    )
    assert all(
        flow.state["structures"]["list"]["works"][str(i)]["calls"] == {CacheCallsKeys.LATEST_CALL_HASH: None}
        for i in range(4)
    )
    assert all(flow.state["structures"]["list"]["works"][str(i)]["changes"] == {} for i in range(4))

    # state_vars
    assert len(flow.state_vars["structures"]["list"]["works"]) == len(flow.list) == 4
    assert list(flow.state_vars["structures"]["list"]["works"].keys()) == ["0", "1", "2", "3"]
    assert all(
        flow.state_vars["structures"]["list"]["works"][str(i)]["vars"]
        == {
            "c": 0,
            "_url": "",
            "_future_url": "",
            "_port": 1,
            "_host": "127.0.0.1",
            "_paths": {},
            "_restarting": False,
            "_internal_ip": "",
            "_public_ip": "",
            "_display_name": "",
            "_cloud_compute": {
                "type": "__cloud_compute__",
                "name": "cpu-small",
                "disk_size": 0,
                "idle_timeout": None,
                "mounts": None,
                "shm_size": 0,
                "_internal_id": "default",
                "interruptible": False,
                "colocation_group_id": None,
            },
        }
        for i in range(4)
    )

    # state_with_changes
    assert len(flow.state_with_changes["structures"]["list"]["works"]) == len(flow.list) == 4
    assert list(flow.state_with_changes["structures"]["list"]["works"].keys()) == ["0", "1", "2", "3"]
    assert all(
        flow.state_with_changes["structures"]["list"]["works"][str(i)]["vars"]
        == {
            "c": 0,
            "_url": "",
            "_future_url": "",
            "_port": 1,
            "_host": "127.0.0.1",
            "_paths": {},
            "_restarting": False,
            "_internal_ip": "",
            "_public_ip": "",
            "_display_name": "",
            "_cloud_compute": {
                "type": "__cloud_compute__",
                "name": "cpu-small",
                "disk_size": 0,
                "idle_timeout": None,
                "mounts": None,
                "shm_size": 0,
                "_internal_id": "default",
                "interruptible": False,
                "colocation_group_id": None,
            },
        }
        for i in range(4)
    )
    assert all(
        flow.state_with_changes["structures"]["list"]["works"][str(i)]["calls"]
        == {CacheCallsKeys.LATEST_CALL_HASH: None}
        for i in range(4)
    )
    assert all(flow.state_with_changes["structures"]["list"]["works"][str(i)]["changes"] == {} for i in range(4))

    # set_state
    state = deepcopy(flow.state)
    state["structures"]["list"]["works"]["0"]["vars"]["c"] = 1
    flow.set_state(state)
    assert flow.list[0].c == 1


def test_list_name():
    lst = List(EmptyFlow(), EmptyFlow())
    assert lst.name == "root"
    assert lst[0].name == "root.0"
    assert lst[1].name == "root.1"

    class RootFlow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.list = List(EmptyFlow(), EmptyFlow())

        def run(self):
            pass

    root = RootFlow()
    assert root.name == "root"
    assert root.list.name == "root.list"
    assert root.list[0].name == "root.list.0"
    assert root.list[1].name == "root.list.1"


class CounterWork(LightningWork):
    def __init__(self, cache_calls, parallel=False):
        super().__init__(cache_calls=cache_calls, parallel=parallel)
        self.counter = 0

    def run(self):
        self.counter += 1


@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut exception")
@pytest.mark.xfail(strict=False, reason="tchaton: Resolve this test.")
@pytest.mark.parametrize("run_once_iterable", [False, True])
@pytest.mark.parametrize("cache_calls", [False, True])
@pytest.mark.parametrize("use_list", [False, True])
def test_structure_with_iterate_and_fault_tolerance(run_once_iterable, cache_calls, use_list):
    class DummyFlow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def run(self):
            pass

    class RootFlow(LightningFlow):
        def __init__(self, use_list, run_once_iterable, cache_calls):
            super().__init__()
            self.looping = 0
            self.run_once_iterable = run_once_iterable
            self.restarting = False
            if use_list:
                self.iter = List(
                    CounterWork(cache_calls),
                    CounterWork(cache_calls),
                    CounterWork(cache_calls),
                    CounterWork(cache_calls),
                    DummyFlow(),
                )
            else:
                self.iter = Dict(**{
                    "0": CounterWork(cache_calls),
                    "1": CounterWork(cache_calls),
                    "2": CounterWork(cache_calls),
                    "3": CounterWork(cache_calls),
                    "4": DummyFlow(),
                })

        def run(self):
            for work_idx, work in self.experimental_iterate(enumerate(self.iter), run_once=self.run_once_iterable):
                if not self.restarting and work_idx == 1:
                    # gives time to the delta to be sent.
                    self.stop()
                if isinstance(work, str) and isinstance(self.iter, Dict):
                    work = self.iter[work]
                work.run()
            if self.looping > 0:
                self.stop()
            self.looping += 1

    app = LightningApp(RootFlow(use_list, run_once_iterable, cache_calls))
    MultiProcessRuntime(app, start_server=False).dispatch()
    assert app.root.iter[0 if use_list else "0"].counter == 1
    assert app.root.iter[1 if use_list else "1"].counter == 0
    assert app.root.iter[2 if use_list else "2"].counter == 0
    assert app.root.iter[3 if use_list else "3"].counter == 0

    app = LightningApp(RootFlow(use_list, run_once_iterable, cache_calls))
    app.root.restarting = True
    MultiProcessRuntime(app, start_server=False).dispatch()

    expected_value = 1 if run_once_iterable else 1 if cache_calls else 2
    assert app.root.iter[0 if use_list else "0"].counter == expected_value
    assert app.root.iter[1 if use_list else "1"].counter == expected_value
    assert app.root.iter[2 if use_list else "2"].counter == expected_value
    assert app.root.iter[3 if use_list else "3"].counter == expected_value


class CheckpointCounter(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False)
        self.counter = 0

    def run(self):
        self.counter += 1


class CheckpointFlow(LightningFlow):
    def __init__(self, collection, depth=0, exit=11):
        super().__init__()
        self.depth = depth
        self.exit = exit
        if depth == 0:
            self.counter = 0

        if depth >= 4:
            self.collection = collection
        else:
            self.flow = CheckpointFlow(collection, depth + 1)

    def run(self):
        if hasattr(self, "counter"):
            self.counter += 1
            if self.counter >= self.exit:
                self.stop()
        if self.depth >= 4:
            self.collection.run()
        else:
            self.flow.run()


class SimpleCounterWork(LightningWork):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        self.counter += 1


class FlowDict(LightningFlow):
    def __init__(self):
        super().__init__()
        self.dict = Dict()

    def run(self):
        if "w" not in self.dict:
            self.dict["w"] = SimpleCounterWork()

        if self.dict["w"].status.stage == WorkStageStatus.SUCCEEDED:
            self.stop()

        self.dict["w"].run()


def test_dict_with_queues():
    app = LightningApp(FlowDict())
    MultiProcessRuntime(app, start_server=False).dispatch()


class FlowList(LightningFlow):
    def __init__(self):
        super().__init__()
        self.list = List()

    def run(self):
        if not len(self.list):
            self.list.append(SimpleCounterWork())

        if self.list[-1].status.stage == WorkStageStatus.SUCCEEDED:
            self.stop()

        self.list[-1].run()


def test_list_with_queues():
    app = LightningApp(FlowList())
    MultiProcessRuntime(app, start_server=False).dispatch()


class WorkS(LightningWork):
    def __init__(self):
        super().__init__()
        self.payload = None

    def run(self):
        self.payload = Payload(2)


class WorkD(LightningWork):
    def run(self, payload):
        assert payload.value == 2


class FlowPayload(LightningFlow):
    def __init__(self):
        super().__init__()
        self.src = WorkS()
        self.dst = Dict(**{"0": WorkD(parallel=True), "1": WorkD(parallel=True)})

    def run(self):
        self.src.run()
        if self.src.payload:
            for work in self.dst.values():
                work.run(self.src.payload)
        if all(w.has_succeeded for w in self.dst.values()):
            self.stop()


@pytest.mark.xfail(strict=False, reason="flaky")
def test_structures_with_payload():
    app = LightningApp(FlowPayload(), log_level="debug")
    MultiProcessRuntime(app, start_server=False).dispatch()
    os.remove("payload")


def test_structures_have_name_on_init():
    """Test that the children in structures have the correct name assigned upon initialization."""

    class ChildWork(LightningWork):
        def run(self):
            pass

    class Collection(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.list_structure = List()
            self.list_structure.append(ChildWork())

            self.dict_structure = Dict()
            self.dict_structure["dict_child"] = ChildWork()

    flow = Collection()
    LightningApp(flow)  # wrap in app to init all component names
    assert flow.list_structure[0].name == "root.list_structure.0"
    assert flow.dict_structure["dict_child"].name == "root.dict_structure.dict_child"


class FlowWiStructures(LightningFlow):
    def __init__(self):
        super().__init__()

        self.ws = [EmptyFlow(), EmptyFlow()]

        self.ws1 = {"a": EmptyFlow(), "b": EmptyFlow()}

        self.ws2 = {
            "a": EmptyFlow(),
            "b": EmptyFlow(),
            "c": List(EmptyFlow(), EmptyFlow()),
            "d": Dict(**{"a": EmptyFlow()}),
        }

    def run(self):
        pass


def test_flow_without_structures():
    flow = FlowWiStructures()
    assert isinstance(flow.ws, List)
    assert isinstance(flow.ws1, Dict)
