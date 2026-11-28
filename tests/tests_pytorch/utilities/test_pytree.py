import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch.utilities._pytree import _tree_flatten, tree_unflatten


def assert_tree_flatten_unflatten(pytree, leaves):
    flat, spec = _tree_flatten(pytree)
    assert flat == leaves
    unflattened = tree_unflatten(flat, spec)
    assert unflattened == pytree


def test_flatten_unflatten():
    dataset1, dataset2 = [0, 1, 2], [0, 1, 2, 3, 4]
    datasets = [[dataset1, dataset2]]
    assert_tree_flatten_unflatten(datasets, [dataset1, dataset2])
    datasets = {"dataset1": [dataset1], "dataset2": dataset2}
    assert_tree_flatten_unflatten(datasets, [dataset1, dataset2])

    dataset1, dataset2 = (0.0, 1.0), (2.0, True)
    datasets = ((dataset1, dataset2),)
    assert_tree_flatten_unflatten(datasets, [dataset1, dataset2])

    dataset1, dataset2 = range(3), range(5)
    datasets = [[dataset1, dataset2]]
    assert_tree_flatten_unflatten(datasets, [dataset1, dataset2])
    datasets = {"datasets": {1: dataset1, 2: [dataset1, dataset2]}}
    assert_tree_flatten_unflatten(datasets, [dataset1, dataset1, dataset2])

    dataset1, dataset2 = torch.randn(2, 3, 2), torch.randn(4, 5, 6)
    datasets = [[dataset1, dataset2]]
    assert_tree_flatten_unflatten(datasets, [dataset1, dataset2])

    dataset1, dataset2 = TensorDataset(dataset1), TensorDataset(dataset2)
    datasets = [[dataset1, dataset2]]
    assert_tree_flatten_unflatten(datasets, [dataset1, dataset2])

    dl1, dl2 = DataLoader(range(3), batch_size=4), DataLoader(range(5), batch_size=5)
    loaders = {"a": dl1, "b": dl2}
    assert_tree_flatten_unflatten(loaders, [dl1, dl2])

    dataset1, dataset2 = ["a", "b"], ["c"]
    datasets = [[dataset1, dataset2]]
    assert_tree_flatten_unflatten(datasets, [dataset1, dataset2])


def test_flatten_unflatten_depth_2_or_more():
    datasets = [range(1), [range(2), [range(3)]]]
    flat, spec = _tree_flatten(datasets)
    assert flat == [range(1), range(2), range(3)]
    unflattened = tree_unflatten(flat, spec)
    assert unflattened == datasets

    datasets = [[1], [[2], [[3]]]]
    flat, spec = _tree_flatten(datasets)
    assert flat == [[1], [2], [3]]
    unflattened = tree_unflatten(flat, spec)
    assert unflattened == datasets

    datasets = [1, [2, [3]]]
    flat, spec = _tree_flatten(datasets)
    # [3] is a container of all primitives so it is treated as a leaf
    assert flat == [1, 2, [3]]
    unflattened = tree_unflatten(flat, spec)
    assert unflattened == datasets


def test_tree_flatten_calls_flatten_fn_once_per_container(monkeypatch):
    from lightning.pytorch.utilities import _pytree as pytree_mod

    orig_node_def = pytree_mod.SUPPORTED_NODES[dict]
    call_count = 0

    def counting_flatten_fn(d):
        nonlocal call_count
        call_count += 1
        return orig_node_def.flatten_fn(d)

    monkeypatch.setitem(
        pytree_mod.SUPPORTED_NODES,
        dict,
        orig_node_def._replace(flatten_fn=counting_flatten_fn)
        if hasattr(orig_node_def, "_replace")
        else type(orig_node_def)(
            counting_flatten_fn,
            orig_node_def.unflatten_fn,
            *((orig_node_def.flatten_with_keys_fn,) if hasattr(orig_node_def, "flatten_with_keys_fn") else ()),
        ),
    )

    nested = {"outer": {"inner": range(3), "prim": [1, 2]}}
    flat, spec = _tree_flatten(nested)
    assert flat == [range(3), [1, 2]]
    assert tree_unflatten(flat, spec) == nested
    assert call_count == 2
