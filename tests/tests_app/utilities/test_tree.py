import pytest
from lightning.app import LightningFlow, LightningWork
from lightning.app.testing.helpers import EmptyFlow, EmptyWork
from lightning.app.utilities.tree import breadth_first


class LeafFlow(EmptyFlow):
    pass


class LeafWork(EmptyWork):
    pass


class SimpleFlowTree(EmptyFlow):
    def __init__(self):
        super().__init__()
        self.simple_flow_left = LeafFlow()
        self.simple_flow_right = LeafFlow()


class SimpleWorkTree(EmptyFlow):
    def __init__(self):
        super().__init__()
        self.simple_work_left = LeafWork()
        self.simple_work_right = LeafWork()


class MixedTree(EmptyFlow):
    def __init__(self):
        super().__init__()
        self.mixed_left = SimpleFlowTree()
        self.work_tree = SimpleWorkTree()
        self.mixed_right = SimpleFlowTree()


@pytest.mark.parametrize(
    ("input_tree", "types", "expected_sequence"),
    [
        (LeafFlow(), (LightningFlow,), ["root"]),
        (LeafWork(), (LightningFlow,), []),
        (
            SimpleFlowTree(),
            (LightningFlow,),
            [
                "root",
                "root.simple_flow_left",
                "root.simple_flow_right",
            ],
        ),
        (SimpleWorkTree(), (LightningFlow,), ["root"]),
        (
            SimpleWorkTree(),
            (LightningFlow, LightningWork),
            [
                "root",
                "root.simple_work_left",
                "root.simple_work_right",
            ],
        ),
        (
            MixedTree(),
            (LightningFlow,),
            [
                "root",
                "root.mixed_left",
                "root.mixed_right",
                "root.work_tree",
                "root.mixed_left.simple_flow_left",
                "root.mixed_left.simple_flow_right",
                "root.mixed_right.simple_flow_left",
                "root.mixed_right.simple_flow_right",
            ],
        ),
        (
            MixedTree(),
            (LightningWork,),
            [
                "root.work_tree.simple_work_left",
                "root.work_tree.simple_work_right",
            ],
        ),
        (
            MixedTree(),
            (LightningFlow, LightningWork),
            [
                "root",
                "root.mixed_left",
                "root.mixed_right",
                "root.work_tree",
                "root.mixed_left.simple_flow_left",
                "root.mixed_left.simple_flow_right",
                "root.mixed_right.simple_flow_left",
                "root.mixed_right.simple_flow_right",
                "root.work_tree.simple_work_left",
                "root.work_tree.simple_work_right",
            ],
        ),
    ],
)
def test_breadth_first(input_tree, types, expected_sequence):
    assert [node.name for node in breadth_first(input_tree, types=types)] == expected_sequence
