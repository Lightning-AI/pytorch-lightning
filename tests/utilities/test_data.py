import pytest
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

from pytorch_lightning.utilities.data import extract_batch_size, get_len, has_iterable_dataset, has_len
from tests.helpers.boring_model import RandomDataset, RandomIterableDataset


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
    assert has_iterable_dataset(DataLoader(RandomIterableDataset(1, 1)))

    assert not has_iterable_dataset(DataLoader(RandomDataset(1, 1)))

    class MockDatasetWithoutIterableDataset(Dataset):
        def __iter__(self):
            yield 1
            return self

    assert not has_iterable_dataset(DataLoader(MockDatasetWithoutIterableDataset()))


def test_has_len():
    assert has_len(DataLoader(RandomDataset(1, 1)))

    with pytest.raises(ValueError, match="`Dataloader` returned 0 length."):
        assert has_len(DataLoader(RandomDataset(0, 0)))

    assert not has_len(DataLoader(RandomIterableDataset(1, 1)))


def test_get_len():
    assert get_len(DataLoader(RandomDataset(1, 1))) == 1

    value = get_len(DataLoader(RandomIterableDataset(1, 1)))

    assert isinstance(value, float)
    assert value == float("inf")
