# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter, DataLoader
from torch.utils.data.dataset import IterableDataset

from pytorch_lightning.utilities.apply_func import apply_to_collection, apply_to_collections
from pytorch_lightning.utilities.auto_restart import (
    _reload_dataloader_state_dict,
    MergedIteratorState,
    patch_dataloader_iterator,
)
from pytorch_lightning.utilities.data import get_len
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training


class TensorRunningAccum:
    """Tracks a running accumulation values (min, max, mean) without graph references.

    Examples:
        >>> accum = TensorRunningAccum(5)
        >>> accum.last(), accum.mean()
        (None, None)
        >>> accum.append(torch.tensor(1.5))
        >>> accum.last(), accum.mean()
        (tensor(1.5000), tensor(1.5000))
        >>> accum.append(torch.tensor(2.5))
        >>> accum.last(), accum.mean()
        (tensor(2.5000), tensor(2.))
        >>> accum.reset()
        >>> _= [accum.append(torch.tensor(i)) for i in range(13)]
        >>> accum.last(), accum.mean(), accum.min(), accum.max()
        (tensor(12.), tensor(10.), tensor(8.), tensor(12.))
    """

    def __init__(self, window_length: int):
        self.window_length = window_length
        self.memory = None
        self.current_idx: int = 0
        self.last_idx: Optional[int] = None
        self.rotated: bool = False

    def reset(self, window_length: Optional[int] = None) -> None:
        """Empty the accumulator."""
        if window_length is None:
            window_length = self.window_length
        self.__init__(window_length)

    def last(self):
        """Get the last added element."""
        if self.last_idx is not None:
            return self.memory[self.last_idx].float()

    def append(self, x):
        """Add an element to the accumulator."""
        if self.memory is None:
            # tradeoff memory for speed by keeping the memory on device
            self.memory = torch.zeros(self.window_length, *x.shape, device=x.device, dtype=x.dtype)

        # store without grads
        with torch.no_grad():
            self.memory[self.current_idx] = x
            self.last_idx = self.current_idx

        # increase index
        self.current_idx += 1

        # reset index when hit limit of tensor
        self.current_idx = self.current_idx % self.window_length
        if self.current_idx == 0:
            self.rotated = True

    def mean(self):
        """Get mean value from stored elements."""
        return self._agg_memory("mean")

    def max(self):
        """Get maximal value from stored elements."""
        return self._agg_memory("max")

    def min(self):
        """Get minimal value from stored elements."""
        return self._agg_memory("min")

    def _agg_memory(self, how: str):
        if self.last_idx is not None:
            if self.rotated:
                return getattr(self.memory.float(), how)()
            return getattr(self.memory[: self.current_idx].float(), how)()


@dataclass
class SharedCycleIteratorState:
    """A state shared between all CylceIterators in a CombinedLoader.

    With a shared state, the iterators can decide to terminate based on the state of all others. If the mode is
    *max_size_cycle*, all iterators need to have finished before the combined loading is considered finished, and
    otherwise any iterator finishing early will lead to all iterators ending early.
    """

    mode: str = "max_size_cycle"
    dataloaders: List[DataLoader] = field(default_factory=lambda: [])
    has_finished: Dict[int, bool] = field(default_factory=lambda: {})
    has_reset: bool = False

    def reset(self) -> None:
        for dataloader in self.dataloaders:
            self.has_finished[id(dataloader)] = False
        self.has_reset = True

    @property
    def done(self) -> bool:
        if not self.has_reset:
            raise MisconfigurationException("Please call reset once all dataloaders have been added.")
        if len(self.dataloaders) == 1:
            return False
        decision_fn = all if self.mode == "max_size_cycle" else any
        return decision_fn(self.has_finished.values())


class CycleIterator:
    """Iterator for restarting a dataloader if it runs out of samples."""

    def __init__(self, loader: Any, length: Optional[int] = None, state: SharedCycleIteratorState = None):
        """
        Args:
            loader: the loader to restart for cyclic (and optionally infinite) sampling
            length: the number of batches to sample (with restarted loaders if necessary) before raising StopIteration
                if None: infinite
        """
        if length is None:
            length = float("inf")

        if not state:
            state = SharedCycleIteratorState()
            state.dataloaders.append(loader)
            state.reset()
        else:
            state.dataloaders.append(loader)

        self.state = state

        self.length = length
        self.loader = loader
        self._loader_iter = None
        self.counter = 0
        self.state = state

    def __iter__(self) -> Any:
        """Creates the internal iterator and returns self.

        Returns:
            CycleIterator: self
        """
        self.counter = 0
        self.state.reset()
        self._loader_iter = iter(self.loader)
        return self

    def __next__(self) -> Any:
        """
        Fetches the next batch from internal dataloader and restarts
        it if necessary
        Returns:
            Any: the resulting batch
        Raises:
            StopIteration: if more then :attr:`length` batches have been returned
        """
        # Note: if self.length is `inf`, then the iterator will never stop
        if self.counter >= self.__len__() or self.state.done:
            raise StopIteration

        try:
            return next(self._loader_iter)

        except StopIteration:

            # inform the shared state this loader has completed
            self.state.has_finished[id(self.loader)] = True

            # check if iteration should be stopped.
            if self.state.done:
                raise StopIteration

            self._loader_iter = iter(self.loader)
            # if fault tolerant is enabled, we need to patch the iterator to collect the states
            # before the batch gets returned.
            fetcher = getattr(self.loader, "_lightning_fetcher", None)
            if fetcher:
                patch_dataloader_iterator(self.loader, self._loader_iter, fetcher)

            return next(self._loader_iter)

        finally:
            self.counter += 1

    def __len__(self) -> Union[int, float]:
        return self.length


class CombinedDataset:
    """Combine multiple datasets and compute their statistics."""

    COMPUTE_FUNCS = {"min_size": min, "max_size_cycle": max}

    def __init__(self, datasets: Union[Sequence, Mapping], mode: str = "min_size"):
        """
        Args:
            datasets: a sequence/mapping datasets. Can be a collections of torch.utils.Dataset,
                Iterable or even None.
            mode: whether to use the minimum number of batches in all samples or the maximum
                number of batches in all samples.
        """
        self.datasets = datasets
        if mode not in self.COMPUTE_FUNCS.keys():
            raise MisconfigurationException(
                f'You have selected unsupported mode "{mode}",'
                f" please select one the: {list(self.COMPUTE_FUNCS.keys())}."
            )
        self.mode = mode

    @property
    def max_len(self) -> Union[int, float]:
        return self._calc_num_data(self.datasets, "max_size_cycle")

    @property
    def min_len(self) -> Union[int, float]:
        return self._calc_num_data(self.datasets, "min_size")

    def _calc_num_data(self, datasets: Union[Sequence, Mapping], mode: str) -> Union[int, float]:
        """Compute the length of `CombinedDataset` according to the `mode`.

        Args:
            datasets: a sequence/mapping datasets. Can be a collections of torch.utils.data.Dataset,
                Iterable or even None.
            mode: Determine `CombinedDataset`'s length is the maximum or minimum of
                the datasets.

        Returns:
            length: the length of `CombinedDataset`
        """
        if mode not in CombinedDataset.COMPUTE_FUNCS.keys():
            raise MisconfigurationException(f"Invalid Mode: {mode}")

        # extract the lengths
        all_lengths = self._get_len_recursive(datasets)

        compute_func = CombinedDataset.COMPUTE_FUNCS[mode]

        if isinstance(all_lengths, (int, float)):
            length = all_lengths
        else:
            length = _nested_calc_num_data(all_lengths, compute_func)

        return length

    def _get_len_recursive(self, data) -> int:
        if isinstance(data, Dataset):
            return len(data)

        if isinstance(data, (float, int)):
            return data

        if isinstance(data, Mapping):
            if any(isinstance(v, (Mapping, Sequence, Dataset, Iterable)) for v in data.values()):
                return {k: self._get_len_recursive(v) for k, v in data.items()}
        elif isinstance(data, Sequence):
            data = list(data)
            if any(isinstance(v, (Mapping, Sequence, Dataset, Iterable)) for v in data):
                return [self._get_len_recursive(v) for v in data]

        return self._get_len(data)

    @staticmethod
    def _get_len(dataset) -> int:
        try:
            return len(dataset)
        except (TypeError, NotImplementedError):
            return float("inf")

    def __len__(self) -> int:
        """Return the minimum length of the datasets."""
        return self._calc_num_data(self.datasets, self.mode)


class CombinedLoader:
    """Combines different dataloaders and allows sampling in parallel. Supported modes are ``"min_size"``, which
    raises StopIteration after the shortest loader (the one with the lowest number of batches) is done, and
    ``"max_size_cycle"`` which raises StopIteration after the longest loader (the one with most batches) is done,
    while cycling through the shorter loaders.

    Examples:
        >>> loaders = {'a': torch.utils.data.DataLoader(range(6), batch_size=4),
        ...            'b': torch.utils.data.DataLoader(range(15), batch_size=5)}
        >>> combined_loader = CombinedLoader(loaders, 'max_size_cycle')
        >>> for item in combined_loader:
        ...     print(item)
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([10, 11, 12, 13, 14])}
        >>> combined_loader = CombinedLoader(loaders, 'min_size')
        >>> for item in combined_loader:
        ...     print(item)
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}
    """

    SUPPORTED_MODES = ("min_size", "max_size_cycle")

    def __init__(self, loaders: Any, mode: str = "min_size"):
        """
        Args:
            loaders: the loaders to sample from. Can be all kind of collection
            mode: the mode. Supported are 'min_size' which stops if the shortest loader is exhausted and
                'max_size_cycle' which stops if the longest loader is exhausted and cycles through the smaller ones.
        """
        if mode not in self.SUPPORTED_MODES:
            raise MisconfigurationException(f"Invalid Mode: {mode}")

        self.loaders = loaders

        datasets = apply_to_collection(
            self.loaders, Iterable, getattr, "dataset", None, wrong_dtype=(Sequence, Mapping)
        )
        # could be multiple datasets, but use self.dataset to follow the name convention in DataLoader
        self.dataset = CombinedDataset(datasets, mode)

        self.mode = mode

        if self.mode == "max_size_cycle":
            self._wrap_loaders_max_size_cycle()

        self._loaders_iter_state_dict = None
        self._iterator = None  # assigned in __iter__

    @staticmethod
    def _state_dict_fn(iterator: Optional[Iterator], has_completed: int) -> Dict:
        if isinstance(iterator, CycleIterator):
            iterator = iterator._loader_iter

        # There is currently 2 dataloader states being tracked: (batch_n - 1, state_n - 1), (batch_n, state_n)
        # where `n` is the current batch. If the batch was processed, it should be saved to reproduce the next batch.
        # Otherwise, we want to get the state of the previous batch, so we can reproduce the current batch.
        # The state is stored directly on the Iterator as an attribute by the DataFetcher for accessibility
        state_to_save = "state" if has_completed else "previous_state"
        state: Optional[MergedIteratorState] = getattr(iterator, state_to_save, None)
        if state:
            return asdict(state)
        return {}

    def state_dict(self, has_completed: bool = False) -> Dict:
        """The state dict includes all states from wrapped dataloaders and their samplers through the
        ``CaptureIterableDataset`` and fast-forward samplers.

        Args:
            has_completed: whether the current state of data fetching is considered completed or not. If it is, the
                current state gets returned, otherwise the previously cached state.
        """
        if not _fault_tolerant_training() or self._iterator is None:
            return {}

        return apply_to_collection(
            self._iterator.loader_iters,
            Iterator,
            self._state_dict_fn,
            has_completed=has_completed,
        )

    def load_state_dict(self, state_dict) -> None:
        # store the samplers state.
        # They would be reloaded once the `CombinedIterator` as been created
        # and the workers are created.
        self._loaders_iter_state_dict = state_dict

    def on_restart(self, iterator: Iterator) -> None:
        if not self._loaders_iter_state_dict:
            return

        def create_loader_iters(dataloader: DataLoader, state_dict: Dict) -> Iterator:
            """Function used to reload the iterator state before once the workers are created."""

            dataloader_to_iter_on = dataloader
            if isinstance(dataloader, CycleIterator):
                dataloader = dataloader_to_iter_on.loader

            # dataset states are collected across all ranks
            rank = torch.distributed.get_rank() if distributed_available() else 0
            state_dict = state_dict[rank]

            _reload_dataloader_state_dict(dataloader, state_dict)

            # We finally spawned the workers if any.
            it = iter(dataloader_to_iter_on)

            # restore caching state
            state = MergedIteratorState.from_state_dict(state_dict)

            if isinstance(dataloader_to_iter_on, CycleIterator):
                it._loader_iter.state = state
            else:
                it.state = state
            return it

        # create an un-existing token, so it doesn't activate for something else than an iterator.
        class DataLoaderDict(dict):
            pass

        # apply the `create_loader_iters` on the collection of `DataLoader / Iterator`.
        # each `Iterator` was created from the `DataLoader`.
        iterator._loader_iters = apply_to_collections(
            self.loaders,
            self._loaders_iter_state_dict,
            (Iterable, DataLoaderDict),
            create_loader_iters,
            wrong_dtype=(Sequence, Mapping),
        )

        self._loaders_iter_state_dict = None

    @property
    def sampler(self) -> Union[Iterable, Sequence, Mapping]:
        """Return a collections of samplers extracting from loaders."""
        return apply_to_collection(self.loaders, (DataLoader, IterableDataset), getattr, "sampler", None)

    def _wrap_loaders_max_size_cycle(self) -> Any:
        """Wraps all loaders to make sure they are cycled until the longest loader is exhausted.

        Returns:
            the wrapped loaders
        """
        all_lengths = apply_to_collection(self.loaders, Iterable, get_len, wrong_dtype=(Sequence, Mapping))

        length = _nested_calc_num_data(all_lengths, max)

        # multiple loaders
        if isinstance(self.loaders, (Sequence, Mapping)):
            state = SharedCycleIteratorState()

            self.loaders = apply_to_collection(
                self.loaders, Iterable, CycleIterator, length=length, state=state, wrong_dtype=(Sequence, Mapping)
            )
            state.reset()

    def _apply_cycle_iterator_length(self) -> None:
        """When the model is `max_size_cycle`, compute the length across all ``CycleIterator`` and re-assign it to
        all dataloaders."""
        if self.mode != "max_size_cycle":
            return

        def set_len(cycle_iterator: CycleIterator, length: int) -> None:
            cycle_iterator.length = length

        all_lengths = apply_to_collection(self.loaders, CycleIterator, lambda c: get_len(c.loader))
        max_length = _nested_calc_num_data(all_lengths, max)
        apply_to_collection(self.loaders, CycleIterator, set_len, length=max_length)

    def __iter__(self) -> Any:
        """Create and return an iterator, `CombinedLoaderIterator`, for the combined loader."""

        # prevent `NotImplementedError` from PyTorch:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/dataloader.py#L541
        def __getstate__patch__(*_):
            return {}

        _BaseDataLoaderIter.__getstate__ = __getstate__patch__
        iterator = CombinedLoaderIterator(self.loaders)
        # handle fault tolerant restart logic.
        self.on_restart(iterator)
        self._iterator = iterator
        return iterator

    @staticmethod
    def _calc_num_batches(loaders: Any, mode="min_size") -> Union[int, float]:
        """Compute the length (aka the number of batches) of `CombinedLoader`.

        Args:
            loaders: a collections of loaders.
            mode: Mode used by the CombinedDataloader

        Returns:
            length: the minimum length of loaders
        """
        all_lengths = apply_to_collection(loaders, Iterable, get_len, wrong_dtype=(Sequence, Mapping))

        if isinstance(all_lengths, (int, float)):
            return all_lengths
        return _nested_calc_num_data(all_lengths, max if mode == "max_size_cycle" else min)

    def __len__(self) -> int:
        return self._calc_num_batches(self.loaders, mode=self.mode)

    @staticmethod
    def _shutdown_workers_and_reset_iterator(dataloader) -> None:
        if hasattr(dataloader, "_iterator") and isinstance(dataloader._iterator, _MultiProcessingDataLoaderIter):
            dataloader._iterator._shutdown_workers()
        dataloader._iterator = None

    def reset(self):
        if self._iterator:
            self._iterator._loader_iters = None
        if self.loaders is not None:
            apply_to_collection(self.loaders, DataLoader, self._shutdown_workers_and_reset_iterator)
        self._iterator = None


class CombinedLoaderIterator:
    """Custom Iterator returning data from multiple loaders, and allows sampling in parallel."""

    def __init__(self, loaders: Any):
        """
        Args:
            loaders: the loaders to sample from. Can be all kind of collection
        """
        self.loaders = loaders
        self._loader_iters = None

    @property
    def loader_iters(self) -> Any:
        """Get the `_loader_iters` and create one if it is None."""
        if self._loader_iters is None:
            self._loader_iters = self.create_loader_iters(self.loaders)

        return self._loader_iters

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        """Fetches the next batch from multiple data loaders.

        Returns:
            a collections of batch data
        """
        return self.request_next_batch(self.loader_iters)

    @staticmethod
    def request_next_batch(loader_iters: Union[Iterator, Sequence, Mapping]) -> Any:
        """Return the batch of data from multiple iterators.

        Args:
            loader_iters: a collections of iterators

        Returns
            Any: a collections of batch data
        """
        return apply_to_collection(loader_iters, Iterator, next)

    @staticmethod
    def create_loader_iters(
        loaders: Union[Any, Iterator, Sequence, Mapping]
    ) -> Union[Any, Iterator, Sequence, Mapping]:
        """Create and return a collection of iterators from loaders.

        Args:
            loaders: a collections of loaders

        Returns
            a collections of iterators
        """
        # dataloaders are Iterable but not Sequences. Need this to specifically exclude sequences
        return apply_to_collection(loaders, Iterable, iter, wrong_dtype=(Sequence, Mapping))


def _nested_calc_num_data(data: Union[Mapping, Sequence], compute_func: Callable):

    if isinstance(data, (float, int)):
        return data

    if isinstance(data, Mapping):
        data = list(data.values())

    if not isinstance(data, Sequence):
        raise TypeError(f"Expected data to be int, Sequence or Mapping, but got {type(data).__name__}")

    new_data = []

    for x in data:
        if isinstance(x, (Mapping, Sequence)):
            new_data.append(_nested_calc_num_data(x, compute_func))
        else:
            new_data.append(x)

    return compute_func(new_data)
