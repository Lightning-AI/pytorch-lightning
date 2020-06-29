from collections import Sequence

import torch

from pytorch_lightning.trainer.supporters import CycleIterator, CombinedLoaderIterator


def test_cycle_iterator():
    iterator = CycleIterator(range(100), 1000)
    assert len(iterator) == 1000
    for idx, item in enumerate(iterator):
        assert item < 100

    assert idx == len(iterator) - 1


def test_combined_loader_iterator_dict_min_size():
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_loader = CombinedLoaderIterator(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_iterator_dict_max_size_cycle():
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_loader = CombinedLoaderIterator(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_combined_loader_iterator_sequence_min_size():
    loaders = [torch.utils.data.DataLoader(range(10), batch_size=4),
               torch.utils.data.DataLoader(range(20), batch_size=5)]

    combined_loader = CombinedLoaderIterator(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


def test_combined_loader_iterator_sequence_max_size_cycle():
    loaders = [torch.utils.data.DataLoader(range(10), batch_size=4),
               torch.utils.data.DataLoader(range(20), batch_size=5)]

    combined_loader = CombinedLoaderIterator(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1
