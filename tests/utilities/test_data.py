import pytest
import torch
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.data import (
    _get_dataloader_init_kwargs,
    _replace_dataloader_init_method,
    _update_dataloader,
    extract_batch_size,
    get_len,
    has_iterable_dataset,
    has_len,
    has_len_all_ranks,
    warning_cache,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDataset, RandomIterableDataset
from tests.helpers.utils import no_warning_call


def test_extract_batch_size():
    """Tests the behavior of extracting the batch size."""

    def _check_warning_not_raised(data, expected):
        with no_warning_call(match="Trying to infer the `batch_size`"):
            assert extract_batch_size(data) == expected

    def _check_warning_raised(data, expected):
        with pytest.warns(UserWarning, match=f"Trying to infer the `batch_size` .* we found is {expected}."):
            assert extract_batch_size(batch) == expected
        warning_cache.clear()

    def _check_error_raised(data):
        with pytest.raises(MisconfigurationException, match="We could not infer the batch_size"):
            extract_batch_size(batch)

    # Warning not raised
    batch = torch.zeros(11, 10, 9, 8)
    _check_warning_not_raised(batch, 11)

    batch = {"test": torch.zeros(11, 10)}
    _check_warning_not_raised(batch, 11)

    batch = [torch.zeros(11, 10)]
    _check_warning_not_raised(batch, 11)

    batch = {"test": [{"test": [torch.zeros(11, 10)]}]}
    _check_warning_not_raised(batch, 11)

    # Warning raised
    batch = {"a": [torch.tensor(1), torch.tensor(2)], "b": torch.tensor([1, 2, 3, 4])}
    _check_warning_raised(batch, 1)

    batch = {"test": [{"test": [torch.zeros(11, 10), torch.zeros(10, 10)]}]}
    _check_warning_raised(batch, 11)

    batch = {"test": [{"test": [torch.zeros(10, 10), torch.zeros(11, 10)]}]}
    _check_warning_raised(batch, 10)

    batch = [{"test": torch.zeros(10, 10), "test_1": torch.zeros(11, 10)}]
    _check_warning_raised(batch, 10)

    # Error raised
    batch = "test string"
    _check_error_raised(batch)

    data = {"test": ["some text"] * 7}
    _check_error_raised(data)

    class CustomBatch:
        def __init__(self):
            self.x = torch.randn(7, 2)

    data = CustomBatch()
    _check_error_raised(data)


def test_has_iterable_dataset():
    assert has_iterable_dataset(DataLoader(RandomIterableDataset(1, 1)))

    assert not has_iterable_dataset(DataLoader(RandomDataset(1, 1)))

    class MockDatasetWithoutIterableDataset(RandomDataset):
        def __iter__(self):
            yield 1
            return self

    assert not has_iterable_dataset(DataLoader(MockDatasetWithoutIterableDataset(1, 1)))


def test_has_len():
    assert has_len(DataLoader(RandomDataset(1, 1)))

    with pytest.warns(UserWarning, match="`DataLoader` returned 0 length."):
        assert has_len(DataLoader(RandomDataset(0, 0)))

    assert not has_len(DataLoader(RandomIterableDataset(1, 1)))


def test_get_len():
    assert get_len(DataLoader(RandomDataset(1, 1))) == 1

    value = get_len(DataLoader(RandomIterableDataset(1, 1)))

    assert isinstance(value, float)
    assert value == float("inf")


def test_has_len_all_rank():
    trainer = Trainer(fast_dev_run=True)
    model = BoringModel()

    with pytest.warns(UserWarning, match="Total length of `DataLoader` across ranks is zero."):
        assert has_len_all_ranks(DataLoader(RandomDataset(0, 0)), trainer.strategy, model)

    assert has_len_all_ranks(DataLoader(RandomDataset(1, 1)), trainer.strategy, model)


def test_update_dataloader_typerror_custom_exception():
    class BadImpl(DataLoader):
        def __init__(self, foo, *args, **kwargs):
            self.foo = foo
            # positional conflict with `dataset`
            super().__init__(foo, *args, **kwargs)

    dataloader = BadImpl([1, 2, 3])
    with pytest.raises(MisconfigurationException, match="`DataLoader` implementation has an error.*`dataset`"):
        _update_dataloader(dataloader, dataloader.sampler)

    class BadImpl2(DataLoader):
        def __init__(self, randomize, *args, **kwargs):
            self.randomize = randomize
            # keyword conflict with `shuffle`
            super().__init__(*args, shuffle=randomize, **kwargs)

    dataloader = BadImpl2(False, [])
    with pytest.raises(MisconfigurationException, match="`DataLoader` implementation has an error.*`shuffle`"):
        _update_dataloader(dataloader, dataloader.sampler)

    class GoodImpl(DataLoader):
        def __init__(self, randomize, *args, **kwargs):
            # fixed implementation, kwargs are filtered
            self.randomize = randomize or kwargs.pop("shuffle", False)
            super().__init__(*args, shuffle=randomize, **kwargs)

    dataloader = GoodImpl(False, [])
    new_dataloader = _update_dataloader(dataloader, dataloader.sampler)
    assert isinstance(new_dataloader, GoodImpl)


def test_replace_dataloader_init_method():
    """Test that context manager intercepts arguments passed to custom subclasses of torch.utils.DataLoader and
    sets them as attributes."""

    class DataLoaderSubclass1(DataLoader):
        def __init__(self, attribute1, *args, **kwargs):
            # intentionally not setting this attribute, calling super with different args
            # self.attribute1 = attribute1
            super().__init__(*args, **kwargs)

    class DataLoaderSubclass2(DataLoaderSubclass1):
        def __init__(self, attribute1, attribute2, *args, **kwargs):
            # intentionally not setting this attribute, calling super with different args
            # self.attribute2 = attribute2
            super().__init__(attribute1, *args, **kwargs)

    with _replace_dataloader_init_method():
        dataloader = DataLoaderSubclass1("attribute1", dataset=range(4), batch_size=2)
        assert dataloader.attribute1 == "attribute1"

    with _replace_dataloader_init_method():
        dataloader = DataLoaderSubclass2("attribute1", "attribute2", dataset=range(4), batch_size=2)
        assert dataloader.attribute1 == "attribute1"
        assert dataloader.attribute2 == "attribute2"

    # `poptorch.DataLoader` uses this pattern, simulate it
    class PoptorchDataLoader(DataLoader):
        def __init__(self, options, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._options = options

        @property
        def options(self):
            return self._options

    # †his read-only property pattern is fine
    dataloader = PoptorchDataLoader(123, [1])
    assert dataloader.options == 123
    # still works with the init replacement
    with _replace_dataloader_init_method():
        dataloader = PoptorchDataLoader(123, [1])
    assert dataloader.options == 123


@pytest.mark.parametrize("mode", [RunningStage.TRAINING, RunningStage.PREDICTING, RunningStage.TESTING])
def test_dataloader_kwargs_replacement_with_iterable_dataset(mode):
    """Test that DataLoader kwargs are not replaced when using Iterable Dataset."""
    dataset = RandomIterableDataset(7, 100)
    dataloader = DataLoader(dataset, batch_size=32)
    dl_kwargs = _get_dataloader_init_kwargs(dataloader, dataloader.sampler, mode=mode)
    assert dl_kwargs["sampler"] is None
    assert dl_kwargs["batch_sampler"] is None
    assert dl_kwargs["batch_size"] is dataloader.batch_size
    assert dl_kwargs["dataset"] is dataloader.dataset
    assert dl_kwargs["collate_fn"] is dataloader.collate_fn
