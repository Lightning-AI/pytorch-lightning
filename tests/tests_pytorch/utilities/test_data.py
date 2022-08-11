import random
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.data import (
    _dataloader_init_kwargs_resolve_sampler,
    _get_dataloader_init_args_and_kwargs,
    _replace_init_method,
    _replace_value_in_saved_args,
    _update_dataloader,
    extract_batch_size,
    get_len,
    has_iterable_dataset,
    has_len,
    has_len_all_ranks,
    warning_cache,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
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

    with _replace_init_method(DataLoader, "dataset"):
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


def test_replace_init_method_multiple_loaders_without_init():
    """In case of a class, that inherits from a class that we are patching, but doesn't define its own `__init__`
    method (the one we are wrapping), it can happen, that `hasattr(cls, "_old_init")` is True because of parent
    class, but it is impossible to delete, because that method is owned by parent class. Furthermore, the error
    occured only sometimes because it depends on the order in which we are iterating over a set of classes we are
    patching.

    This test simulates the behavior by generating sufficient number of dummy classes, which do not define `__init__`
    and are children of `DataLoader`. We are testing that a) context manager `_replace_init_method` exits cleanly, and
    b) the mechanism checking for presence of `_old_init` works as expected.
    """
    classes = [DataLoader]
    for i in range(100):
        classes.append(type(f"DataLoader_{i}", (random.choice(classes),), {}))

    with _replace_init_method(DataLoader, "dataset"):
        for cls in classes[1:]:  # First one is `DataLoader`
            assert "_old_init" not in cls.__dict__
            assert hasattr(cls, "_old_init")

        assert "_old_init" in DataLoader.__dict__
        assert hasattr(DataLoader, "_old_init")


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
def test_replace_init_method_dataloader(cls, args, kwargs, arg_names, dataset, checked_values):
    with _replace_init_method(DataLoader, "dataset"):
        dataloader = cls(*args, **kwargs)

    assert dataloader.__pl_saved_args == args
    assert dataloader.__pl_saved_kwargs == kwargs
    assert dataloader.__pl_saved_arg_names == arg_names
    assert dataloader.__pl_saved_default_kwargs == {}
    assert dataloader.__dataset == dataset

    assert dataloader.dataset == dataset

    for key, value in checked_values.items():
        dataloader_value = getattr(dataloader, key)
        if isinstance(dataloader_value, torch.Tensor):
            assert dataloader_value is value
        else:
            assert dataloader_value == value

    dataloader = _update_dataloader(dataloader, dataloader.sampler)

    assert isinstance(dataloader, cls)
    assert not hasattr(dataloader, "__pl_saved_kwargs")
    assert not hasattr(dataloader, "__pl_saved_arg_names")
    assert not hasattr(dataloader, "__pl_saved_args")
    assert not hasattr(dataloader, "__pl_saved_default_kwargs")
    assert not hasattr(dataloader, "__dataset")

    assert dataloader.dataset == dataset

    for key, value in checked_values.items():
        dataloader_value = getattr(dataloader, key)
        if isinstance(dataloader_value, torch.Tensor):
            assert dataloader_value is value
        else:
            assert dataloader_value == value


def test_replace_init_method_extra_kwargs():
    class LoaderSubclass(DataLoader):
        def __init__(self, dataset, *args, batch_size=10, **kwargs):
            super().__init__(dataset, *args, batch_size=batch_size, **kwargs)

    with _replace_init_method(DataLoader, "dataset"):
        dataloader = LoaderSubclass(range(10))

    assert dataloader.__pl_saved_args == (range(10),)
    assert dataloader.__pl_saved_kwargs == {}
    assert dataloader.__pl_saved_arg_names == ("dataset",)
    assert dataloader.__pl_saved_default_kwargs == {"batch_size": 10}
    assert dataloader.__dataset == range(10)


@pytest.mark.parametrize("predicting", [True, False])
def test_custom_batch_sampler(predicting):
    """This test asserts, that custom `BatchSampler`, with all the arguments, that are required in order to
    properly reinstantiate the class, is invoked properly.

    It also asserts, that during the reinstantiation, the wrapper of `__init__` method is not present anymore, therefore
    not setting `__pl_saved_{args,arg_names,kwargs}` attributes.
    """

    class MyBatchSampler(BatchSampler):
        # Custom Batch sampler with extra argument and default value
        def __init__(self, sampler, extra_arg, drop_last=True):
            self.extra_arg = extra_arg
            super().__init__(sampler, 10, drop_last)

    sampler = RandomSampler(range(10))
    with _replace_init_method(BatchSampler):
        # instantiate within `_replace_init_method` context manager, simulating `*_dataloader` hooks
        batch_sampler = MyBatchSampler(sampler, "random_str")

    dataloader = DataLoader(range(10), batch_sampler=batch_sampler)

    # assert that passed information got saved
    assert dataloader.batch_sampler.__pl_saved_args == (sampler, "random_str")
    assert dataloader.batch_sampler.__pl_saved_kwargs == {}
    assert dataloader.batch_sampler.__pl_saved_arg_names == ("sampler", "extra_arg")
    assert dataloader.batch_sampler.__pl_saved_default_kwargs == {"drop_last": True}

    # updating dataloader, what happens on access of the dataloaders.
    # This should not fail, and would fail before support for custom args.
    dataloader = _update_dataloader(
        dataloader, dataloader.sampler, mode=RunningStage.PREDICTING if predicting else None
    )

    # Assert the `__init__` method is not replaced anymore and everything is instantiated to correct types
    batch_sampler = dataloader.batch_sampler

    if predicting:
        assert isinstance(batch_sampler, IndexBatchSamplerWrapper)
        batch_sampler = batch_sampler._sampler

    assert isinstance(batch_sampler, MyBatchSampler)
    assert batch_sampler.drop_last == (not predicting)

    assert batch_sampler.extra_arg == "random_str"
    assert not hasattr(batch_sampler, "__pl_saved_kwargs")
    assert not hasattr(batch_sampler, "__pl_saved_arg_names")
    assert not hasattr(batch_sampler, "__pl_saved_args")
    assert not hasattr(batch_sampler, "__pl_saved_default_kwargs")


def test_custom_batch_sampler_no_drop_last():
    """Tests whether appropriate warning is raised when the custom `BatchSampler` does not support `drop_last` and
    we want to reset it."""

    class MyBatchSampler(BatchSampler):
        # Custom batch sampler with extra argument, but without `drop_last`
        def __init__(self, sampler, extra_arg):
            self.extra_arg = extra_arg
            super().__init__(sampler, 10, False)

    sampler = RandomSampler(range(10))
    with _replace_init_method(BatchSampler):
        # instantiate within `_replace_init_method` context manager, simulating `*_dataloader` hooks
        batch_sampler = MyBatchSampler(sampler, "random_str")

    dataloader = DataLoader(range(10), batch_sampler=batch_sampler)

    # assert that passed information got saved
    assert dataloader.batch_sampler.__pl_saved_args == (sampler, "random_str")
    assert dataloader.batch_sampler.__pl_saved_kwargs == {}
    assert dataloader.batch_sampler.__pl_saved_arg_names == ("sampler", "extra_arg")
    assert dataloader.batch_sampler.__pl_saved_default_kwargs == {}

    # Assert that warning is raised
    with pytest.warns(UserWarning, match="drop_last=False"):
        dataloader = _update_dataloader(dataloader, dataloader.sampler, mode=RunningStage.PREDICTING)


def test_custom_batch_sampler_no_sampler():
    """Tests whether appropriate error is raised when the custom `BatchSampler` does not support sampler
    argument."""

    class MyBatchSampler(BatchSampler):
        # Custom batch sampler, without sampler argument.
        def __init__(self, extra_arg):
            self.extra_arg = extra_arg
            super().__init__(RandomSampler(range(10)), 10, False)

    with _replace_init_method(BatchSampler):
        # instantiate within `_replace_init_method` context manager, simulating `*_dataloader` hooks
        batch_sampler = MyBatchSampler("random_str")
    dataloader = DataLoader(range(10), batch_sampler=batch_sampler)

    # assert that passed information got saved
    assert dataloader.batch_sampler.__pl_saved_args == ("random_str",)
    assert dataloader.batch_sampler.__pl_saved_kwargs == {}
    assert dataloader.batch_sampler.__pl_saved_arg_names == ("extra_arg",)
    assert dataloader.batch_sampler.__pl_saved_default_kwargs == {}

    # Assert that error is raised
    with pytest.raises(TypeError, match="sampler into the batch sampler"):
        dataloader = _update_dataloader(dataloader, dataloader.sampler, mode=RunningStage.PREDICTING)


@pytest.mark.parametrize(
    [
        "args",
        "kwargs",
        "default_kwargs",
        "arg_names",
        "replace_key",
        "replace_value",
        "expected_status",
        "expected_args",
        "expected_kwargs",
    ],
    [
        pytest.param((), {}, {}, [], "a", 1, False, (), {}, id="empty"),
        pytest.param((1,), {}, {}, ["a"], "a", 2, True, (2,), {}, id="simple1"),
        pytest.param((1, 2, 3), {}, {}, ["a", "b", "c"], "b", False, True, (1, False, 3), {}, id="simple2"),
        pytest.param((1, 2, 3), {"a": 1}, {}, ["b", "c", "d"], "a", 2, True, (1, 2, 3), {"a": 2}, id="simple_kwargs"),
        pytest.param(
            (1, 2, 3),
            {"a": 1},
            {"e": 5},
            ["b", "c", "d"],
            "e",
            2,
            True,
            (1, 2, 3),
            {"a": 1, "e": 2},
            id="default_kwargs",
        ),
    ],
)
def test_replace_value_in_args(
    args, kwargs, default_kwargs, arg_names, replace_key, replace_value, expected_status, expected_args, expected_kwargs
):
    assert _replace_value_in_saved_args(replace_key, replace_value, args, kwargs, default_kwargs, arg_names) == (
        expected_status,
        expected_args,
        expected_kwargs,
    )


def test_dataloader_disallow_batch_sampler():
    dataset = RandomDataset(5, 100)
    dataloader = DataLoader(dataset, batch_size=10)

    # This should not raise
    _dataloader_init_kwargs_resolve_sampler(dataloader, dataloader.sampler, disallow_batch_sampler=True)

    dataset = RandomDataset(5, 100)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=10, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    # this should raise - using batch sampler, that was not automatically instantiated by DataLoader
    with pytest.raises(MisconfigurationException, match="when running on multiple IPU devices"):
        _dataloader_init_kwargs_resolve_sampler(dataloader, dataloader.sampler, disallow_batch_sampler=True)


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
