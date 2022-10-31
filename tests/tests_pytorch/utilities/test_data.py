from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from lightning_lite.utilities.data import _replace_dunder_methods
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.data import (
    _dataloader_init_kwargs_resolve_sampler,
    _get_dataloader_init_args_and_kwargs,
    _update_dataloader,
    extract_batch_size,
    get_len,
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
    with _replace_dunder_methods(BatchSampler):
        # instantiate within `_replace_dunder_method` context manager, simulating `*_dataloader` hooks
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
        dataloader = _update_dataloader(dataloader, dataloader.sampler, mode=RunningStage.PREDICTING)


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
