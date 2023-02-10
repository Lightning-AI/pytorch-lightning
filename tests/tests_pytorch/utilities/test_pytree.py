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
