from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from lightning.fabric.utilities.data import _replace_dunder_methods
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import RandomDataset, RandomIterableDataset
from lightning.pytorch.overrides.distributed import _IndexBatchSamplerWrapper
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.data import (
    _get_dataloader_init_args_and_kwargs,
    _update_dataloader,
    extract_batch_size,
    has_len_all_ranks,
    warning_cache,
)
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning_utilities.test.warning import no_warning_call
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, RandomSampler


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


def test_has_len_all_rank():
    trainer = Trainer(fast_dev_run=True)

    with pytest.warns(UserWarning, match="Total length of `DataLoader` across ranks is zero."):
        assert has_len_all_ranks(DataLoader(RandomDataset(0, 0)), trainer.strategy)

    assert has_len_all_ranks(DataLoader(RandomDataset(1, 1)), trainer.strategy)


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
def test_custom_torch_batch_sampler(predicting):
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
    dataloader = _update_dataloader(
        dataloader, dataloader.sampler, mode=(RunningStage.PREDICTING if predicting else None)
    )

    # Assert the `__init__` method is not replaced anymore and everything is instantiated to correct types
    batch_sampler = dataloader.batch_sampler

    if predicting:
        assert isinstance(batch_sampler, _IndexBatchSamplerWrapper)
        batch_sampler = batch_sampler._batch_sampler

    assert isinstance(batch_sampler, MyBatchSampler)
    assert batch_sampler.drop_last == (not predicting)

    assert batch_sampler.extra_arg == "random_str"
    assert not hasattr(batch_sampler, "__pl_saved_kwargs")
    assert not hasattr(batch_sampler, "__pl_saved_arg_names")
    assert not hasattr(batch_sampler, "__pl_saved_args")
    assert not hasattr(batch_sampler, "__pl_saved_default_kwargs")


@pytest.mark.parametrize("predicting", [True, False])
def test_custom_torch_batch_sampler_doppelganger(predicting):
    """Test we can reinstantiate a sampler that mimics PyTorch's BatchSampler even if it does not inherit from it.

    This is only possible if that sampler accepts the `batch_size` and `drop_last` arguments, and stores them
    as attributes.

    """

    class BatchSamplerDoppelganger:
        """A batch sampler that mimics `torch.utils.data.BatchSampler` but does not inherit from it."""

        def __init__(self, sampler, batch_size, drop_last):
            self.sampler = sampler
            self.batch_size = batch_size
            self.drop_last = drop_last

        def __iter__(self):
            while True:
                yield [0, 1, 2, 3]

        def __len__(self) -> int:
            return 4

    batch_sampler = BatchSamplerDoppelganger(sampler=Mock(), batch_size=2, drop_last=True)
    dataloader = DataLoader(range(100), batch_sampler=batch_sampler)
    new_sampler = Mock()
    dataloader = _update_dataloader(
        dataloader, sampler=new_sampler, mode=(RunningStage.PREDICTING if predicting else None)
    )

    batch_sampler = dataloader.batch_sampler

    if predicting:
        assert isinstance(batch_sampler, _IndexBatchSamplerWrapper)
        batch_sampler = batch_sampler._batch_sampler

    assert isinstance(batch_sampler, BatchSamplerDoppelganger)
    assert batch_sampler.sampler == new_sampler
    assert batch_sampler.drop_last == (not predicting)


@pytest.mark.parametrize("predicting", [True, False])
def test_custom_batch_sampler(predicting):
    """Test that a custom (non-PyTorch) batch sampler requires the user to set `use_distributed_sampler=False`."""

    class CustomBatchSampler:  # not inheriting from `BatchSampler`
        def __iter__(self):
            while True:
                yield [0, 1, 2, 3]

    batch_sampler = CustomBatchSampler()
    dataloader = DataLoader(range(100), batch_sampler=batch_sampler)

    if predicting:
        with pytest.warns(PossibleUserWarning, match=r"Make sure your sampler is configured correctly to return all"):
            _ = _update_dataloader(dataloader, sampler=Mock(), mode=RunningStage.PREDICTING)
    else:
        with pytest.raises(TypeError, match=r"can't inject a \(distributed\) sampler into your batch sampler"):
            _ = _update_dataloader(dataloader, sampler=Mock(), mode=None)


def test_custom_batch_sampler_no_drop_last():
    """Tests whether appropriate warning is raised when the custom `BatchSampler` does not support `drop_last` and we
    want to reset it."""

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
        _ = _update_dataloader(dataloader, dataloader.sampler, mode=RunningStage.PREDICTING)


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
        _ = _update_dataloader(dataloader, dataloader.sampler, mode=RunningStage.PREDICTING)


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
    _, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, dataloader.sampler)
    assert dl_kwargs["indices"] is dataloader.indices
