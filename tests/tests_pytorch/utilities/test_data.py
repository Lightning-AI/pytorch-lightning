from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel, RandomDataset
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.data import (
    _get_dataloader_init_args_and_kwargs,
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
from tests_pytorch.helpers.datasets import RandomIterableDataset
from tests_pytorch.helpers.utils import no_warning_call


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

    @dataclass
    class CustomDataclass:
        a: Tensor
        b: Tensor

    # Warning not raised
    batch = torch.zeros(11, 10, 9, 8)
    _check_warning_not_raised(batch, 11)

    batch = {"test": torch.zeros(11, 10)}
    _check_warning_not_raised(batch, 11)

    batch = [torch.zeros(11, 10)]
    _check_warning_not_raised(batch, 11)

    batch = CustomDataclass(torch.zeros(11, 10), torch.zeros(11, 10))
    _check_warning_not_raised(batch, 11)

    batch = {"test": [{"test": [torch.zeros(11, 10)]}]}
    _check_warning_not_raised(batch, 11)

    # Warning raised
    batch = {"a": [torch.tensor(1), torch.tensor(2)], "b": torch.tensor([1, 2, 3, 4])}
    _check_warning_raised(batch, 1)

    batch = CustomDataclass(torch.zeros(11, 10), torch.zeros(1))
    _check_warning_raised(batch, 11)

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
    class BadStandaloneGoodHookImpl(DataLoader):
        def __init__(self, foo, *args, **kwargs):
            self.foo = foo
            # positional conflict with `dataset`
            super().__init__(foo, *args, **kwargs)

    dataloader = BadStandaloneGoodHookImpl([1, 2, 3])
    with pytest.raises(MisconfigurationException, match="`DataLoader` implementation has an error.*`dataset`"):
        _update_dataloader(dataloader, dataloader.sampler)

    with _replace_dataloader_init_method():
        dataloader = BadStandaloneGoodHookImpl([1, 2, 3])
    new_dataloader = _update_dataloader(dataloader, dataloader.sampler)
    assert isinstance(new_dataloader, BadStandaloneGoodHookImpl)

    class BadImpl(DataLoader):
        def __init__(self, randomize, *args, **kwargs):
            self.randomize = randomize
            # keyword conflict with `shuffle`
            super().__init__(*args, shuffle=randomize, **kwargs)

    dataloader = BadImpl(False, [])
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


class DataLoaderSubclass1(DataLoader):
    def __init__(self, attribute1, *args, **kwargs):
        self.at1 = attribute1
        super().__init__(*args, **kwargs)


class DataLoaderSubclass2(DataLoaderSubclass1):
    def __init__(self, attribute2, *args, **kwargs):
        self.at2 = attribute2
        super().__init__(attribute2 + "-2", *args, **kwargs)


class MyBaseDataLoader(DataLoader):
    pass


class MyDataLoader(MyBaseDataLoader):
    def __init__(self, data: torch.Tensor, *args, **kwargs):
        self.data = data
        super().__init__(range(data.size(0)), *args, **kwargs)


test3_data = torch.randn((10, 20))


class PoptorchDataLoader(DataLoader):
    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._options = options

    @property
    def options(self):
        return self._options


class IncompleteDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, **kwargs):
        batch_size = max(batch_size - 5, 0)
        super().__init__(dataset, batch_size=batch_size, **kwargs)


class WeirdDataLoader1(DataLoader):
    def __init__(self, arg1, arg2, **kwargs):
        self.arg1 = arg1
        super().__init__(arg2, **kwargs)


class WeirdDataLoader2(DataLoader):
    def __init__(self, data_part1, data_part2, **kwargs):
        data = list(data_part1) + list(data_part2)
        super().__init__(data, **kwargs)


class NoneDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ChangingDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(list(dataset) + list(range(5, 10)), **kwargs)


@pytest.mark.parametrize(
    ["cls", "args", "kwargs", "arg_names", "dataset", "checked_values"],
    [
        pytest.param(
            DataLoaderSubclass1,
            ("attribute1",),
            dict(dataset=range(4), batch_size=2),
            ("attribute1",),
            range(4),
            dict(batch_size=2, at1="attribute1"),
            id="test1",
        ),
        pytest.param(
            DataLoaderSubclass2,
            ("attribute2",),
            dict(dataset=range(4), batch_size=2),
            ("attribute2",),
            range(4),
            dict(batch_size=2, at1="attribute2-2", at2="attribute2"),
            id="test2",
        ),
        pytest.param(
            MyDataLoader,
            (test3_data,),
            dict(batch_size=2),
            ("data",),
            range(10),
            dict(batch_size=2, data=test3_data),
            id="test3",
        ),
        pytest.param(PoptorchDataLoader, (123, [1]), dict(), ("options",), [1], dict(options=123), id="test4"),
        pytest.param(
            IncompleteDataLoader,
            (range(10),),
            dict(batch_size=10),
            ("dataset",),
            range(10),
            dict(batch_size=5),
            id="test5",
        ),
        pytest.param(
            WeirdDataLoader1,
            (10, range(10)),
            dict(batch_size=10),
            ("arg1", "arg2"),
            range(10),
            dict(arg1=10, batch_size=10),
            id="test6",
        ),
        pytest.param(
            WeirdDataLoader2,
            (range(10), range(10, 20)),
            dict(batch_size=10),
            ("data_part1", "data_part2"),
            list(range(20)),
            dict(batch_size=10),
            id="test7",
        ),
        pytest.param(NoneDataLoader, (None,), dict(), (), None, dict(), id="test8"),
        pytest.param(ChangingDataLoader, (range(5),), dict(), ("dataset",), list(range(10)), dict(), id="test9"),
    ],
)
def test_replace_dataloader_init_method(cls, args, kwargs, arg_names, dataset, checked_values):
    with _replace_dataloader_init_method():
        dataloader = cls(*args, **kwargs)

    assert dataloader.__pl_dl_args == args
    assert dataloader.__pl_dl_kwargs == kwargs
    assert dataloader.__pl_dl_arg_names == arg_names
    assert dataloader.__dataset == dataset

    assert dataloader.dataset == dataset

    for key, value in checked_values.items():
        dataloader_value = getattr(dataloader, key)
        if isinstance(dataloader_value, torch.Tensor):
            assert dataloader_value is value
        else:
            assert getattr(dataloader, key) == value

    dataloader = _update_dataloader(dataloader, dataloader.sampler)

    assert isinstance(dataloader, cls)
    assert not hasattr(dataloader, "__pl_dl_kwargs")
    assert not hasattr(dataloader, "__pl_dl_arg_names")
    assert not hasattr(dataloader, "__pl_dl_args")
    assert not hasattr(dataloader, "__dataset")

    assert dataloader.dataset == dataset

    for key, value in checked_values.items():
        dataloader_value = getattr(dataloader, key)
        if isinstance(dataloader_value, torch.Tensor):
            assert dataloader_value is value
        else:
            assert getattr(dataloader, key) == value


@pytest.mark.parametrize("mode", [RunningStage.TRAINING, RunningStage.PREDICTING, RunningStage.TESTING])
def test_dataloader_kwargs_replacement_with_iterable_dataset(mode):
    """Test that DataLoader kwargs are not replaced when using Iterable Dataset."""
    dataset = RandomIterableDataset(7, 100)
    dataloader = DataLoader(dataset, batch_size=32)
    _, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, dataloader.sampler, mode=mode)
    assert dl_kwargs["sampler"] is None
    assert dl_kwargs["batch_sampler"] is None
    assert dl_kwargs["batch_size"] is dataloader.batch_size
    assert dl_kwargs["dataset"] is dataloader.dataset
    assert dl_kwargs["collate_fn"] is dataloader.collate_fn
