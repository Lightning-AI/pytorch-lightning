import pytest
from lightning.app.storage.path import Path
from lightning.app.testing.helpers import EmptyFlow, EmptyWork
from lightning.app.utilities.component import (
    _context,
    _convert_paths_after_init,
    _get_context,
    _is_flow_context,
    _is_work_context,
    _set_context,
    _set_work_context,
)
from lightning.app.utilities.enum import ComponentContext


def test_convert_paths_after_init():
    """Test that we can convert paths after the Flow/Work initialization, i.e., when the LightningApp is fully
    instantiated."""

    # TODO: Add a test case for the Lightning List and Dict containers

    class Flow1(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.path1 = Path("a")
            self.path2 = Path("b")

    flow1 = Flow1()
    assert flow1._paths == {}
    _convert_paths_after_init(flow1)
    assert flow1._paths == {"path1": Path("a").to_dict(), "path2": Path("b").to_dict()}

    class Work1(EmptyWork):
        def __init__(self):
            super().__init__()
            self.path3 = Path("c")

    class Flow2(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.work1 = Work1()
            self.path4 = Path("d")

    flow2 = Flow2()
    assert flow2._paths == {}
    assert flow2.work1._paths == {}
    _convert_paths_after_init(flow2)
    assert flow2._paths == {"path4": Path("d").to_dict()}
    assert set(flow2.work1._paths.keys()) == {"path3"}
    assert flow2.work1._paths["path3"]["origin_name"] == "root.work1"
    assert flow2.work1._paths["path3"]["consumer_name"] == "root.work1"


@pytest.mark.parametrize("ctx", [c.value for c in ComponentContext])
def test_context_context_manager(ctx):
    with _context("flow"):
        assert _get_context().value == "flow"
    assert _get_context() is None


@pytest.mark.parametrize("ctx", [c.value for c in ComponentContext])
def test_set_get_context(ctx):
    assert _get_context() is None
    _set_context(ctx)
    assert _get_context().value == ctx


def test_is_context():
    _set_context("flow")
    assert _is_flow_context()

    _set_work_context()
    assert _is_work_context()

    _set_context(None)
    assert not _is_flow_context()
    assert not _is_work_context()
