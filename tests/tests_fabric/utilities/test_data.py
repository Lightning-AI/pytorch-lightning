import contextlib
import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from lightning.fabric.utilities.data import (
    _dataloader_init_kwargs_resolve_sampler,
    _get_dataloader_init_args_and_kwargs,
    _replace_dunder_methods,
    _replace_value_in_saved_args,
    _set_sampler_epoch,
    _update_dataloader,
    _WrapAttrTag,
    has_iterable_dataset,
    has_len,
)
from lightning.fabric.utilities.exceptions import MisconfigurationException
from tests_fabric.helpers.models import RandomDataset, RandomIterableDataset


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


def test_replace_dunder_methods_multiple_loaders_without_init():
    """In case of a class, that inherits from a class that we are patching, but doesn't define its own `__init__`
    method (the one we are wrapping), it can happen, that `hasattr(cls, "__old__init__")` is True because of parent
    class, but it is impossible to delete, because that method is owned by parent class. Furthermore, the error occured
    only sometimes because it depends on the order in which we are iterating over a set of classes we are patching.

    This test simulates the behavior by generating sufficient number of dummy classes, which do not define `__init__`
    and are children of `DataLoader`. We are testing that a) context manager `_replace_dunder_method` exits cleanly, and
    b) the mechanism checking for presence of `__old__init__` works as expected.

    """
    classes = [DataLoader]
    for i in range(100):
        classes.append(type(f"DataLoader_{i}", (random.choice(classes),), {}))

    before = {cls: cls.__init__ for cls in classes}

    with _replace_dunder_methods(DataLoader, "dataset"):
        for cls in classes[1:]:  # First one is `DataLoader`
            assert "__old__init__" not in cls.__dict__
            assert hasattr(cls, "__old__init__")

        assert "__old__init__" in DataLoader.__dict__
        assert hasattr(DataLoader, "__old__init__")

    for cls in classes:
        assert before[cls] == cls.__init__


class MyBaseDataLoader(DataLoader):
    pass


class DataLoaderSubclass1(DataLoader):
    def __init__(self, attribute1, *args, **kwargs):
        self.at1 = attribute1
        super().__init__(*args, **kwargs)


class DataLoaderSubclass2(DataLoaderSubclass1):
    def __init__(self, attribute2, *args, **kwargs):
        self.at2 = attribute2
        super().__init__(attribute2 + "-2", *args, **kwargs)


class MyDataLoader(MyBaseDataLoader):
    def __init__(self, data: Tensor, *args, **kwargs):
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
    ("cls", "args", "kwargs", "arg_names", "dataset", "checked_values"),
    [
        pytest.param(
            DataLoaderSubclass1,
            ("attribute1",),
            {"dataset": range(4), "batch_size": 2},
            ("attribute1",),
            range(4),
            {"batch_size": 2, "at1": "attribute1"},
            id="test1",
        ),
        pytest.param(
            DataLoaderSubclass2,
            ("attribute2",),
            {"dataset": range(4), "batch_size": 2},
            ("attribute2",),
            range(4),
            {"batch_size": 2, "at1": "attribute2-2", "at2": "attribute2"},
            id="test2",
        ),
        pytest.param(
            MyDataLoader,
            (test3_data,),
            {"batch_size": 2},
            ("data",),
            range(10),
            {"batch_size": 2, "data": test3_data},
            id="test3",
        ),
        pytest.param(PoptorchDataLoader, (123, [1]), {}, ("options",), [1], {"options": 123}, id="test4"),
        pytest.param(
            IncompleteDataLoader,
            (range(10),),
            {"batch_size": 10},
            ("dataset",),
            range(10),
            {"batch_size": 5},
            id="test5",
        ),
        pytest.param(
            WeirdDataLoader1,
            (10, range(10)),
            {"batch_size": 10},
            ("arg1", "arg2"),
            range(10),
            {"arg1": 10, "batch_size": 10},
            id="test6",
        ),
        pytest.param(
            WeirdDataLoader2,
            (range(10), range(10, 20)),
            {"batch_size": 10},
            ("data_part1", "data_part2"),
            list(range(20)),
            {"batch_size": 10},
            id="test7",
        ),
        pytest.param(NoneDataLoader, (None,), {}, (), None, {}, id="test8"),
        pytest.param(ChangingDataLoader, (range(5),), {}, ("dataset",), list(range(10)), {}, id="test9"),
    ],
)
def test_replace_dunder_methods_dataloader(cls, args, kwargs, arg_names, dataset, checked_values):
    with _replace_dunder_methods(DataLoader, "dataset"):
        dataloader = cls(*args, **kwargs)

    assert dataloader.__pl_saved_args == args
    assert dataloader.__pl_saved_kwargs == kwargs
    assert dataloader.__pl_saved_arg_names == arg_names
    assert dataloader.__pl_saved_default_kwargs == {}
    assert dataloader.__dataset == dataset

    assert dataloader.dataset == dataset

    for key, value in checked_values.items():
        dataloader_value = getattr(dataloader, key)
        if isinstance(dataloader_value, Tensor):
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
        if isinstance(dataloader_value, Tensor):
            assert dataloader_value is value
        else:
            assert dataloader_value == value


def test_replace_dunder_methods_extra_kwargs():
    class LoaderSubclass(DataLoader):
        def __init__(self, dataset, *args, batch_size=10, **kwargs):
            super().__init__(dataset, *args, batch_size=batch_size, **kwargs)

    with _replace_dunder_methods(DataLoader, "dataset"):
        dataloader = LoaderSubclass(range(10))

    assert dataloader.__pl_saved_args == (range(10),)
    assert dataloader.__pl_saved_kwargs == {}
    assert dataloader.__pl_saved_arg_names == ("dataset",)
    assert dataloader.__pl_saved_default_kwargs == {"batch_size": 10}
    assert dataloader.__dataset == range(10)


def test_replace_dunder_methods_attrs():
    """This test checks, that all the calls from setting and deleting attributes within `_replace_dunder_methods` are
    correctly preserved even after reinstantiation.

    It also includes a custom `__setattr__`

    """

    class Loader(DataLoader):
        def __setattr__(self, attr, val):
            if attr == "custom_arg":
                val = val + 2
            super().__setattr__(attr, val)

    with _replace_dunder_methods(DataLoader, "dataset"):
        dataloader = Loader(range(10))
        dataloader.custom_arg = 5
        dataloader.my_arg = 10
        dataloader.another_arg = 100
        del dataloader.dataset
        with contextlib.suppress(AttributeError):
            del dataloader.abc_arg

    assert dataloader.__pl_saved_args == (range(10),)
    assert dataloader.__pl_saved_kwargs == {}
    assert dataloader.__pl_saved_arg_names == ("dataset",)
    assert dataloader.__dataset == range(10)
    assert dataloader.custom_arg == 7
    assert dataloader.my_arg == 10
    assert dataloader.another_arg == 100
    assert not hasattr(dataloader, "dataset")
    assert dataloader.__pl_attrs_record == [
        (("custom_arg", 5), _WrapAttrTag.SET),
        (("my_arg", 10), _WrapAttrTag.SET),
        (("another_arg", 100), _WrapAttrTag.SET),
        (("dataset",), _WrapAttrTag.DEL),
    ]

    dataloader = _update_dataloader(dataloader, dataloader.sampler)
    assert dataloader.custom_arg == 7
    assert dataloader.my_arg == 10
    assert dataloader.another_arg == 100
    assert not hasattr(dataloader, "dataset")


def test_replace_dunder_methods_restore_methods():
    """This tests checks whether are all dunder methods restored to their original versions."""

    class Init(DataLoader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class SetAttr(DataLoader):
        def __setattr__(self, *args):
            return super().__setattr__(*args)

    class DelAttr(DataLoader):
        def __delattr__(self, *args):
            return super().__delattr__(*args)

    class InitAndSetAttr(Init, SetAttr):
        pass

    class InitAndDelAttr(Init, DelAttr):
        pass

    class SetAttrAndDelAttr(SetAttr, DelAttr):
        pass

    class AllDunder(Init, SetAttr, DelAttr):
        pass

    before = {}
    for cls in (Init, SetAttr, DelAttr, InitAndSetAttr, InitAndDelAttr, SetAttrAndDelAttr, AllDunder):
        before[cls] = {"init": cls.__init__, "setattr": cls.__setattr__, "delattr": cls.__delattr__}

    with _replace_dunder_methods(DataLoader, "dataset"):
        pass

    for cls in (Init, SetAttr, DelAttr, InitAndSetAttr, InitAndDelAttr, SetAttrAndDelAttr, AllDunder):
        assert before[cls] == {"init": cls.__init__, "setattr": cls.__setattr__, "delattr": cls.__delattr__}


@pytest.mark.parametrize(
    (
        "args",
        "kwargs",
        "default_kwargs",
        "arg_names",
        "replace_key",
        "replace_value",
        "expected_status",
        "expected_args",
        "expected_kwargs",
    ),
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


def test_update_dataloader_typerror_custom_exception():
    class BadStandaloneGoodHookImpl(DataLoader):
        def __init__(self, foo, *args, **kwargs):
            self.foo = foo
            # positional conflict with `dataset`
            super().__init__(foo, *args, **kwargs)

    dataloader = BadStandaloneGoodHookImpl([1, 2, 3])
    with pytest.raises(MisconfigurationException, match="implementation has an error.*`dataset`"):
        _update_dataloader(dataloader, dataloader.sampler)

    with _replace_dunder_methods(DataLoader, "dataset"):
        dataloader = BadStandaloneGoodHookImpl([1, 2, 3])
    new_dataloader = _update_dataloader(dataloader, dataloader.sampler)
    assert isinstance(new_dataloader, BadStandaloneGoodHookImpl)

    class BadImpl(DataLoader):
        def __init__(self, randomize, *args, **kwargs):
            self.randomize = randomize
            # keyword conflict with `shuffle`
            super().__init__(*args, shuffle=randomize, **kwargs)

    dataloader = BadImpl(False, [])
    with pytest.raises(MisconfigurationException, match="implementation has an error.*`shuffle`"):
        _update_dataloader(dataloader, dataloader.sampler)

    class GoodImpl(DataLoader):
        def __init__(self, randomize, *args, **kwargs):
            # fixed implementation, kwargs are filtered
            self.randomize = randomize or kwargs.pop("shuffle", False)
            super().__init__(*args, shuffle=randomize, **kwargs)

    dataloader = GoodImpl(False, [])
    new_dataloader = _update_dataloader(dataloader, dataloader.sampler)
    assert isinstance(new_dataloader, GoodImpl)


def test_custom_batch_sampler():
    """This test asserts, that custom `BatchSampler`, with all the arguments, that are required in order to properly
    reinstantiate the class, is invoked properly.

    It also asserts, that during the reinstantiation, the wrapper of `__init__` method is not present anymore, therefore
    not setting `__pl_saved_{args,arg_names,kwargs}` attributes.

    """

    class MyBatchSampler(BatchSampler):
        # Custom Batch sampler with extra argument and default value
        def __init__(self, sampler, extra_arg, drop_last=True):
            self.extra_arg = extra_arg
            super().__init__(sampler, 10, drop_last)

    sampler = RandomSampler(range(10))
    with _replace_dunder_methods(BatchSampler):
        # instantiate within `_replace_dunder_method` context manager, simulating `*_dataloader` hooks
        batch_sampler = MyBatchSampler(sampler, "random_str")

    dataloader = DataLoader(range(10), batch_sampler=batch_sampler)

    # assert that passed information got saved
    assert dataloader.batch_sampler.__pl_saved_args == (sampler, "random_str")
    assert dataloader.batch_sampler.__pl_saved_kwargs == {}
    assert dataloader.batch_sampler.__pl_saved_arg_names == ("sampler", "extra_arg")
    assert dataloader.batch_sampler.__pl_saved_default_kwargs == {"drop_last": True}

    # updating dataloader, what happens on access of the dataloaders.
    # This should not fail, and would fail before support for custom args.
    dataloader = _update_dataloader(dataloader, dataloader.sampler)

    # Assert the `__init__` method is not replaced anymore and everything is instantiated to correct types
    batch_sampler = dataloader.batch_sampler

    assert isinstance(batch_sampler, MyBatchSampler)

    assert batch_sampler.extra_arg == "random_str"
    assert not hasattr(batch_sampler, "__pl_saved_kwargs")
    assert not hasattr(batch_sampler, "__pl_saved_arg_names")
    assert not hasattr(batch_sampler, "__pl_saved_args")
    assert not hasattr(batch_sampler, "__pl_saved_default_kwargs")


def test_custom_batch_sampler_no_sampler():
    """Tests whether appropriate error is raised when the custom `BatchSampler` does not support sampler argument."""

    class MyBatchSampler(BatchSampler):
        # Custom batch sampler, without sampler argument.
        def __init__(self, extra_arg):
            self.extra_arg = extra_arg
            super().__init__(RandomSampler(range(10)), 10, False)

    with _replace_dunder_methods(BatchSampler):
        # instantiate within `_replace_dunder_method` context manager, simulating `*_dataloader` hooks
        batch_sampler = MyBatchSampler("random_str")
    dataloader = DataLoader(range(10), batch_sampler=batch_sampler)

    # assert that passed information got saved
    assert dataloader.batch_sampler.__pl_saved_args == ("random_str",)
    assert dataloader.batch_sampler.__pl_saved_kwargs == {}
    assert dataloader.batch_sampler.__pl_saved_arg_names == ("extra_arg",)
    assert dataloader.batch_sampler.__pl_saved_default_kwargs == {}

    # Assert that error is raised
    with pytest.raises(TypeError, match="sampler into the batch sampler"):
        dataloader = _update_dataloader(dataloader, dataloader.sampler)


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


def test_dataloader_kwargs_replacement_with_iterable_dataset():
    """Test that DataLoader kwargs are not replaced when using Iterable Dataset."""
    dataset = RandomIterableDataset(7, 100)
    dataloader = DataLoader(dataset, batch_size=32)
    _, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, dataloader.sampler)
    assert dl_kwargs["sampler"] is None
    assert dl_kwargs["batch_sampler"] is None
    assert dl_kwargs["batch_size"] is dataloader.batch_size
    assert dl_kwargs["dataset"] is dataloader.dataset
    assert dl_kwargs["collate_fn"] is dataloader.collate_fn


def test_dataloader_kwargs_replacement_with_array_default_comparison():
    """Test that the comparison of attributes and default argument values works with arrays (truth value ambiguous).

    Regression test for issue #15408.

    """
    dataset = RandomDataset(5, 100)

    class ArrayAttributeDataloader(DataLoader):
        def __init__(self, indices=None, **kwargs):
            super().__init__(dataset)
            self.indices = np.random.rand(2, 2)  # an attribute we can't compare with ==

    dataloader = ArrayAttributeDataloader(dataset)
    dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, dataloader.sampler)
    assert dl_kwargs["indices"] is dataloader.indices


def test_set_sampler_epoch():
    # No samplers
    dataloader = Mock()
    dataloader.sampler = None
    dataloader.batch_sampler = None
    _set_sampler_epoch(dataloader, 55)

    # set_epoch not callable
    dataloader = Mock()
    dataloader.sampler.set_epoch = None
    dataloader.batch_sampler.set_epoch = None
    _set_sampler_epoch(dataloader, 55)

    # set_epoch callable
    dataloader = Mock()
    _set_sampler_epoch(dataloader, 55)
    dataloader.sampler.set_epoch.assert_called_once_with(55)
    dataloader.batch_sampler.sampler.set_epoch.assert_called_once_with(55)
