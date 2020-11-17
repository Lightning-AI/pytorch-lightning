from collections import Sequence

import pytest
import torch

from torch.utils.data import TensorDataset
from pytorch_lightning.trainer.train_loader_patch import MultiIterator


def test_multi_iterator_dict_min_size():
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_loader = MultiIterator(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_multi_iterator_dict_max_size_cycle():
    loaders = {'a': torch.utils.data.DataLoader(range(10), batch_size=4),
               'b': torch.utils.data.DataLoader(range(20), batch_size=5)}

    combined_loader = MultiIterator(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders.values()])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, dict)
        assert len(item) == 2
        assert 'a' in item and 'b' in item

    assert idx == len(combined_loader) - 1


def test_multi_iterator_sequence_min_size():
    loaders = [torch.utils.data.DataLoader(range(10), batch_size=4),
               torch.utils.data.DataLoader(range(20), batch_size=5)]

    combined_loader = MultiIterator(loaders, 'min_size')

    assert len(combined_loader) == min([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1


def test_multi_iterator_sequence_max_size_cycle():
    loaders = [torch.utils.data.DataLoader(range(10), batch_size=4),
               torch.utils.data.DataLoader(range(20), batch_size=5)]

    combined_loader = MultiIterator(loaders, 'max_size_cycle')

    assert len(combined_loader) == max([len(v) for v in loaders])

    for idx, item in enumerate(combined_loader):
        assert isinstance(item, Sequence)
        assert len(item) == 2

    assert idx == len(combined_loader) - 1
