import os
import pickle
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from time import time
from unittest.mock import ANY

import pytest
from deepdiff import DeepDiff, Delta

from lightning_app import LightningApp
from lightning_app.core.flow import LightningFlow
from lightning_app.core.work import LightningWork
from lightning_app.runners import MultiProcessRuntime, SingleProcessRuntime
from lightning_app.storage import Path
from lightning_app.storage.path import storage_root_dir
from lightning_app.testing.helpers import EmptyFlow, EmptyWork
from lightning_app.utilities.app_helpers import _delta_to_appstate_delta, _LightningAppRef
from lightning_app.utilities.exceptions import ExitAppException


def test_empty_component():
    class A(LightningFlow):
        def run(self):
            pass

    empty_component = A()
    assert empty_component.state == {
        "vars": {"_layout": ANY, "_paths": {}},
        "calls": {},
        "flows": {},
        "structures": {},
        "changes": {},
        "works": {},
    }


@dataclass
class CustomDataclass:
    x: int = 1
    y: tuple = (3, 2, 1)


@pytest.mark.parametrize(
    "attribute",
    (
        {3, 2, 1},
        lambda _: 5,
        CustomDataclass(),
    ),
)
@pytest.mark.parametrize("cls", (LightningWork, LightningFlow))
def test_unsupported_attribute_types(cls, attribute):
    class Component(cls):
        def __init__(self):
            super().__init__()
            self.x = attribute

        def run(self):
            pass

    with pytest.raises(AttributeError, match="Only JSON-serializable attributes are currently supported"):
        Component()


@pytest.mark.parametrize(
    "name,value",
    [
        ("x", 1),
        ("f", EmptyFlow()),
        ("w", EmptyWork()),
    ],
)
def test_unsupported_attribute_declaration_outside_init_or_run(name, value):
    """Test that LightningFlow attributes (with a few exceptions) are not allowed to be declared outside __init__."""
    flow = EmptyFlow()
    with pytest.raises(AttributeError, match=f"Cannot set attributes that were not defined in __init__: {name}"):
        setattr(flow, name, value)
    assert not hasattr(flow, name)
    assert name not in flow.state["vars"]
    assert name not in flow._works
    assert name not in flow._flows

    # no error for protected attributes, since they don't contribute to the state
    setattr(flow, "_" + name, value)
    assert hasattr(flow, "_" + name)


@pytest.mark.parametrize(
    "name,value",
    [
        ("x", 1),
        ("f", EmptyFlow()),
        ("w", EmptyWork()),
    ],
)
@pytest.mark.parametrize("defined", [False, True])
def test_unsupported_attribute_declaration_inside_run(defined, name, value):
    """Test that LightningFlow attributes can set LightningFlow or LightningWork inside its run method, but everything
    else needs to be defined in the __init__ method."""

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            if defined:
                setattr(self, name, None)

        def run(self):
            if not defined and not isinstance(value, (LightningFlow, LightningWork)):
                with pytest.raises(
                    AttributeError, match=f"Cannot set attributes that were not defined in __init__: {name}"
                ):
                    setattr(self, name, value)
                assert name not in self.state["vars"]
                assert name not in self._works
                assert name not in self._flows
            else:
                setattr(self, name, value)
                if isinstance(value, LightningFlow):
                    assert name in self._flows
                elif isinstance(value, LightningWork):
                    assert name in self._works
                else:
                    assert name in self.state["vars"]

    flow = Flow()
    flow.run()


@pytest.mark.parametrize("value", [EmptyFlow(), EmptyWork()])
def test_name_gets_removed_from_state_when_defined_as_flow_works(value):
    """Test that LightningFlow attributes are removed from the state."""

    class EmptyFlow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.value = None

        def run(self):
            self.value = value

    flow = EmptyFlow()
    flow.run()
    if isinstance(value, LightningFlow):
        assert "value" not in flow.state["vars"]
        assert "value" in flow._flows
    else:
        assert "value" not in flow.state["vars"]
        assert "value" in flow._works


@pytest.mark.parametrize(
    "name,value",
    [
        ("_name", "name"),
        ("_changes", {"change": 1}),
    ],
)
def test_supported_attribute_declaration_outside_init(name, value):
    """Test the custom LightningFlow setattr implementation for the few reserved attributes that are allowed to be set
    from outside __init__."""
    flow = EmptyFlow()
    setattr(flow, name, value)
    assert getattr(flow, name) == value


def test_supported_attribute_declaration_inside_init():
    """Test that the custom LightningFlow setattr can identify the __init__ call in the stack frames above."""

    class Flow(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.directly_in_init = "init"
            self.method_under_init()

        def method_under_init(self):
            self.attribute = "test"
            self.subflow = EmptyFlow()

    flow = Flow()
    assert flow.directly_in_init == "init"
    assert flow.state["vars"]["directly_in_init"] == "init"
    assert flow.attribute == "test"
    assert flow.state["vars"]["attribute"] == "test"
    assert isinstance(flow.subflow, EmptyFlow)
    assert flow.state["flows"]["subflow"] == flow.subflow.state


def test_setattr_outside_run_context():
    """Test that it is allowed to update attributes outside `run` as long as the attribute is already declared."""

    class Flow(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.attribute = ""

        def outside_run(self):
            # reading allowed, setting not allowed
            self.attribute = "allowed"
            return super().configure_layout()

    flow = Flow()
    flow.outside_run()
    assert flow.attribute == "allowed"
    assert flow.state["vars"]["attribute"] == "allowed"


def _run_state_transformation(tmpdir, attribute, update_fn, inplace=False):
    """This helper function defines a flow, assignes an attribute and performs a transformation on the state."""

    class StateTransformationTest(LightningFlow):
        def __init__(self):
            super().__init__()
            self.x = attribute
            self.finished = False

        def run(self):
            if self.finished:
                self._exit()

            x = update_fn(self.x)
            if not inplace:
                self.x = x
            self.finished = True

    flow = StateTransformationTest()
    assert flow.x == attribute
    app = LightningApp(flow)
    SingleProcessRuntime(app, start_server=False).dispatch()
    return app.state["vars"]["x"]


@pytest.mark.parametrize(
    "attribute,update_fn,expected",
    (
        (1, lambda x: x + 1, 2),
        (0.5, lambda x: x + 0.5, 1.0),
        (True, lambda x: not x, False),
        ("cocofruit", lambda x: x + "s", "cocofruits"),
        (dict(a=1, b=2), lambda x: dict(a=1, b=3), dict(a=1, b=3)),
        ([1, 2], lambda x: [1, 2, 3], [1, 2, 3]),
        ((4, 5), lambda x: (4, 5, 6), (4, 5, 6)),
    ),
)
def test_attribute_state_change(attribute, update_fn, expected, tmpdir):
    """Test that state changes get recored on all supported data types."""
    assert _run_state_transformation(tmpdir, attribute, update_fn, inplace=False) == expected


def test_inplace_attribute_state_change(tmpdir):
    """Test that in-place modifications on containers get captured as a state change."""
    # inplace modification of a nested dict
    def transform(x):
        x["b"]["c"] += 1

    value = dict(a=1, b=dict(c=2))
    expected = dict(a=1, b=dict(c=3))
    assert _run_state_transformation(tmpdir, value, transform, inplace=True) == expected

    # inplace modification of nested list
    def transform(x):
        x[2].append(3.0)

    value = ["a", 1, [2.0]]
    expected = ["a", 1, [2.0, 3.0]]
    assert _run_state_transformation(tmpdir, value, transform, inplace=True) == expected

    # inplace modification of a custom dict
    def transform(x):
        x.update("baa")

    value = Counter("abab")
    expected = Counter(a=4, b=3)
    assert _run_state_transformation(tmpdir, value, transform, inplace=True) == expected


def test_lightning_flow_and_work():
    class Work(LightningWork):
        def __init__(self, cache_calls: bool = True, port=None):
            super().__init__(cache_calls=cache_calls, port=port)
            self.counter = 0

        def run(self, *args, **kwargs):
            self.counter += 1

    class Flow_A(LightningFlow):
        def __init__(self):
            super().__init__()
            self.counter = 0
            self.work_a = Work(cache_calls=True, port=8000)
            self.work_b = Work(cache_calls=False, port=8001)

        def run(self):
            if self.counter < 5:
                self.work_a.run()
                self.work_b.run()
                self.counter += 1
            else:
                self._exit()

    flow_a = Flow_A()
    assert flow_a.named_works() == [("work_a", flow_a.work_a), ("work_b", flow_a.work_b)]
    assert flow_a.works() == [flow_a.work_a, flow_a.work_b]
    state = {
        "vars": {"counter": 0, "_layout": ANY, "_paths": {}},
        "calls": {},
        "flows": {},
        "structures": {},
        "works": {
            "work_b": {
                "vars": {
                    "counter": 0,
                    "_url": "",
                    "_future_url": "",
                    "_port": 8001,
                    "_host": "127.0.0.1",
                    "_paths": {},
                    "_restarting": False,
                    "_internal_ip": "",
                },
                "calls": {"latest_call_hash": None},
                "changes": {},
            },
            "work_a": {
                "vars": {
                    "counter": 0,
                    "_url": "",
                    "_future_url": "",
                    "_port": 8000,
                    "_host": "127.0.0.1",
                    "_paths": {},
                    "_restarting": False,
                    "_internal_ip": "",
                },
                "calls": {"latest_call_hash": None},
                "changes": {},
            },
        },
        "changes": {},
    }
    assert flow_a.state == state
    try:
        while True:
            flow_a.run()
    except ExitAppException:
        pass

    state = {
        "vars": {"counter": 5, "_layout": ANY, "_paths": {}},
        "calls": {},
        "flows": {},
        "structures": {},
        "works": {
            "work_b": {
                "vars": {
                    "counter": 5,
                    "_url": "",
                    "_future_url": "",
                    "_port": 8001,
                    "_host": "127.0.0.1",
                    "_paths": {},
                    "_restarting": False,
                    "_internal_ip": "",
                },
                "calls": {"latest_call_hash": None},
                "changes": {},
            },
            "work_a": {
                "vars": {
                    "counter": 1,
                    "_url": "",
                    "_future_url": "",
                    "_port": 8000,
                    "_host": "127.0.0.1",
                    "_paths": {},
                    "_restarting": False,
                    "_internal_ip": "",
                },
                "calls": {
                    "latest_call_hash": None,
                    "run:fe3fa0f34fc1317e152e5afb023332995392071046f1ea51c34c7c9766e3676c": {
                        "name": "run",
                        "call_hash": "run:fe3fa0f34fc1317e152e5afb023332995392071046f1ea51c34c7c9766e3676c",
                        "ret": None,
                    },
                },
                "changes": {},
            },
        },
        "changes": {},
    }
    assert flow_a.state == state


def test_populate_changes():
    class WorkA(LightningWork):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def run(self):
            pass

    class A(LightningFlow):
        def __init__(self):
            super().__init__()
            self.work = WorkA()

        def run(self):
            pass

    flow_a = A()
    flow_state = flow_a.state
    work_state = flow_a.work.state
    flow_a.work.counter = 1
    work_state_2 = flow_a.work.state
    delta = Delta(DeepDiff(work_state, work_state_2))
    delta = _delta_to_appstate_delta(flow_a, flow_a.work, delta)
    new_flow_state = LightningApp.populate_changes(flow_state, flow_state + delta)
    flow_a.set_state(new_flow_state)
    assert flow_a.work.counter == 1
    assert new_flow_state["works"]["work"]["changes"] == {"counter": {"from": 0, "to": 1}}
    assert flow_a.work._changes == {"counter": {"from": 0, "to": 1}}


def test_populate_changes_status_removed():
    """Regression test for https://github.com/PyTorchLightning/lightning/issues/342."""
    last_state = {
        "vars": {},
        "calls": {},
        "flows": {},
        "works": {
            "work": {
                "vars": {},
                "calls": {
                    "latest_call_hash": "run:fe3f",
                    "run:fe3f": {
                        "statuses": [
                            {"stage": "requesting", "message": None, "reason": None, "timestamp": 1},
                            {"stage": "starting", "message": None, "reason": None, "timestamp": 2},
                            {"stage": "requesting", "message": None, "reason": None, "timestamp": 3},
                        ],
                    },
                },
                "changes": {},
            },
        },
        "changes": {},
    }
    new_state = deepcopy(last_state)
    call = new_state["works"]["work"]["calls"]["run:fe3f"]
    call["statuses"] = call["statuses"][:-1]  # pretend that a status was removed from the list
    new_state_before = deepcopy(new_state)
    new_state = LightningApp.populate_changes(last_state, new_state)
    assert new_state == new_state_before


class CFlow(LightningFlow):
    def __init__(self, run_once):
        super().__init__()
        self.looping = 0
        self.tracker = 0
        self.restarting = False
        self.run_once = run_once

    def run(self):
        for idx in self.experimental_iterate(range(0, 10), run_once=self.run_once):
            if not self.restarting and (idx + 1) == 5:
                _LightningAppRef.get_current()._dump_checkpoint()
                self._exit()
            self.tracker += 1
        self.looping += 1
        if self.looping == 2:
            self._exit()


@pytest.mark.parametrize("runtime_cls", [SingleProcessRuntime])
@pytest.mark.parametrize("run_once", [False, True])
def test_lightning_flow_iterate(tmpdir, runtime_cls, run_once):
    app = LightningApp(CFlow(run_once))
    runtime_cls(app, start_server=False).dispatch()
    assert app.root.looping == 0
    assert app.root.tracker == 4
    call_hash = list(v for v in app.root._calls if "experimental_iterate" in v)[0]
    iterate_call = app.root._calls[call_hash]
    assert iterate_call["counter"] == 4
    assert not iterate_call["has_finished"]

    checkpoint_dir = os.path.join(storage_root_dir(), "checkpoints")
    app = LightningApp(CFlow(run_once))
    app.load_state_dict_from_checkpoint_dir(checkpoint_dir)
    app.root.restarting = True
    assert app.root.looping == 0
    assert app.root.tracker == 4
    runtime_cls(app, start_server=False).dispatch()
    assert app.root.looping == 2
    assert app.root.tracker == 10 if run_once else 20
    iterate_call = app.root._calls[call_hash]
    assert iterate_call["has_finished"]


class FlowCounter(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        if self.counter >= 3:
            self._exit()
        self.counter += 1


@pytest.mark.parametrize("runtime_cls", [SingleProcessRuntime, MultiProcessRuntime])
def test_lightning_flow_counter(runtime_cls, tmpdir):

    app = LightningApp(FlowCounter())
    app.checkpointing = True
    runtime_cls(app, start_server=False).dispatch()
    assert app.root.counter == 3

    checkpoint_dir = os.path.join(storage_root_dir(), "checkpoints")
    checkpoints = os.listdir(checkpoint_dir)
    assert len(checkpoints) == 4
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        with open(checkpoint_path, "rb") as f:
            app = LightningApp(FlowCounter())
            app.set_state(pickle.load(f))
            runtime_cls(app, start_server=False).dispatch()
            assert app.root.counter == 3


def test_flow_iterate_method():
    class Flow(LightningFlow):
        def run(self):
            pass

    flow = Flow()
    with pytest.raises(TypeError, match="An iterable should be provided"):
        next(flow.experimental_iterate(1))


def test_flow_path_assignment():
    """Test that paths in the lit format lit:// get converted to a proper lightning_app.storage.Path object."""

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.no_path = "a/b/c"
            self.path = Path("lit://x/y/z")
            self.lit_path = "lit://x/y/z"

    flow = Flow()
    assert isinstance(flow.no_path, str)
    assert isinstance(flow.path, Path)
    assert isinstance(flow.lit_path, Path)
    assert flow.path == flow.lit_path


def test_flow_state_change_with_path():
    """Test that type changes to a Path attribute are properly reflected within the state."""

    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.none_to_path = None
            self.path_to_none = Path()
            self.path_to_path = Path()

        def run(self):
            self.none_to_path = "lit://none/to/path"
            self.path_to_none = None
            self.path_to_path = "lit://path/to/path"
            self._exit()

    flow = Flow()
    MultiProcessRuntime(LightningApp(flow)).dispatch()
    assert flow.none_to_path == Path("lit://none/to/path")
    assert flow.path_to_none is None
    assert flow.path_to_path == Path("lit://path/to/path")

    assert "path_to_none" not in flow._paths
    assert "path_to_none" in flow._state
    assert flow._paths["none_to_path"] == Path("lit://none/to/path").to_dict()
    assert flow._paths["path_to_path"] == Path("lit://path/to/path").to_dict()
    assert flow.state["vars"]["none_to_path"] == Path("lit://none/to/path")
    assert flow.state["vars"]["path_to_none"] is None
    assert flow.state["vars"]["path_to_path"] == Path("lit://path/to/path")


class FlowSchedule(LightningFlow):
    def __init__(self):
        super().__init__()
        self._last_time = None

    def run(self):
        if self.schedule("* * * * * 0,5,10,15,20,25,30,35,40,45,50,55"):
            if self._last_time is None:
                self._last_time = False
            elif not self._last_time:
                self._last_time = time()
            else:
                # TODO (tchaton) Optimize flow execution.
                assert 4.0 < abs(time() - self._last_time) < 6.0
                self._exit()


def test_scheduling_api():

    app = LightningApp(FlowSchedule())
    MultiProcessRuntime(app).dispatch()


def test_lightning_flow():
    class Flow(LightningFlow):
        def run(self):
            if self.schedule("midnight"):
                pass
            if self.schedule("hourly"):
                pass
            if self.schedule("@hourly"):
                pass
            if self.schedule("daily"):
                pass
            if self.schedule("weekly"):
                pass
            if self.schedule("monthly"):
                pass
            if self.schedule("yearly"):
                pass
            if self.schedule("annually"):
                pass
            assert len(self._calls["scheduling"]) == 8

    Flow().run()
