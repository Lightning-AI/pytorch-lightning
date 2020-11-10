from collections import Sequence

import pytest
import torch

from torch.utils.data import TensorDataset
from pytorch_lightning.trainer.supporters import CycleIterator, CombinedLoader, CombinedDataset, CombinedLoaderIterator


def test_cycle_iterator():
    iterator = CycleIterator(range(100), 1000)
    assert len(iterator) == 1000
    for idx, item in enumerate(iterator):
        assert item < 100

    assert idx == len(iterator) - 1


@pytest.mark.parametrize(['dataset_1', 'dataset_2'], [
    ([list(range(10)), list(range(20))]),
    ([range(10), range(20)]),
    ([torch.randn(10, 3, 2), torch.randn(20, 5, 6)]),
    ([TensorDataset(torch.randn(10, 3, 2)), TensorDataset(torch.randn(20, 5, 6))])
])
def test_combined_dataset(dataset_1, dataset_2):
    datasets = [dataset_1, dataset_2]
    combined_dataset = CombinedDataset(datasets)

    assert combined_dataset.max_len == 20
    assert combined_dataset.min_len == len(combined_dataset) == 10


def test_combined_loader_iterator_dict_min_size():
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_iter = CombinedLoaderIterator(loaders)

    for idx, item in enumerate(combined_iter):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == min(len(loaders['a']), len(loaders['b'])) - 1


def test_combined_loader_dict_min_size():
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_loader = CombinedLoader(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_dict_max_size_cycle():
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_loader = CombinedLoader(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_min_size():
    loaders = [torch.utils.data.DataLoader(range(10), batch_size=4),
               torch.utils.data.DataLoader(range(20), batch_size=5)]

    combined_loader = CombinedLoader(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


def test_combined_loader_sequence_max_size_cycle():
    loaders = [torch.utils.data.DataLoader(range(10), batch_size=4),
               torch.utils.data.DataLoader(range(20), batch_size=5)]

    combined_loader = CombinedLoader(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1
