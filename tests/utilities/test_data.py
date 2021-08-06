import pytest
import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

from pytorch_lightning.utilities.data import extract_batch_size, has_iterable_dataset, has_len


def test_extract_batch_size():
    """Tests the behavior of extracting the batch size."""
    batch = "test string"
    assert extract_batch_size(batch) == 11

    batch = torch.zeros(11, 10, 9, 8)
    assert extract_batch_size(batch) == 11

    batch = {"test": torch.zeros(11, 10)}
    assert extract_batch_size(batch) == 11

    batch = [torch.zeros(11, 10)]
    assert extract_batch_size(batch) == 11

    batch = {"test": [{"test": [torch.zeros(11, 10)]}]}
    assert extract_batch_size(batch) == 11


def test_has_iterable_dataset():
    class MockIterableDataset(IterableDataset):
        def __iter__(self):
            yield 1
            return self

    assert has_iterable_dataset(DataLoader(MockIterableDataset()))

    assert not has_iterable_dataset(DataLoader(Dataset()))

    class MockDataset(Dataset):
        def __iter__(self):
            yield 1
            return self

    assert not has_iterable_dataset(DataLoader(MockDataset()))


def test_has_len():
    class MockDatasetWithLen(Dataset):
        def __len__(self):
            return 1

    assert has_len(DataLoader(MockDatasetWithLen()))

    class MockDatasetWithZeroLen(Dataset):
        def __len__(self):
            return 0

    with pytest.raises(ValueError):
        assert has_len(DataLoader(MockDatasetWithZeroLen()))

    assert not has_len(DataLoader(IterableDataset()))
