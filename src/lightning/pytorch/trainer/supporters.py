# Copyright The Lightning AI team.
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
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from lightning_utilities.core.apply_func import apply_to_collection
from torch.utils.data import Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter, DataLoader
from torch.utils.data.dataset import IterableDataset
from typing_extensions import TypedDict

from lightning.fabric.utilities.data import sized_len
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import _NUMBER


@dataclass
class SharedCycleIteratorState:
    """A state shared between all CycleIterators in a CombinedLoader.

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

    def __init__(self, loader: Any, length: _NUMBER = float("inf"), state: SharedCycleIteratorState = None):
        """
        Args:
            loader: the loader to restart for cyclic (and optionally infinite) sampling
            length: the number of batches to sample (with restarted loaders if necessary) before raising StopIteration
        """
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
        assert isinstance(self._loader_iter, Iterator)

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
            return next(self._loader_iter)

        finally:
            self.counter += 1

    def __len__(self) -> _NUMBER:
        # TODO: returning float here is a hack
        return self.length


class _CombinationMode(TypedDict):
    name: str
    fn: Callable[[_NUMBER, _NUMBER], _NUMBER]
    default: _NUMBER


_supported_modes = {
    "min_size": _CombinationMode(name="min_size", fn=min, default=float("inf")),
    "max_size_cycle": _CombinationMode(name="max_size_cycle", fn=max, default=float("-inf")),
}


class CombinedDataset:
    """Combine multiple datasets."""

    def __init__(self, datasets: Any, mode: str = "min_size"):
        """
        Args:
            datasets: Collections of Iterables.
            mode: whether to use the minimum number of batches in all samples or the maximum
                number of batches in all samples.
        """
        if mode not in _supported_modes:
            raise ValueError(f"Unsupported mode {mode!r}, please select one of: {list(_supported_modes)}.")
        self._mode = mode
        self._datasets = datasets

    @property
    def datasets(self) -> Any:
        return self._datasets

    def _get_len_recursive(self, data: Any) -> Union[int, List, Dict]:
        if isinstance(data, int):
            return data
        if isinstance(data, Mapping):
            if any(isinstance(v, (Mapping, Sequence, Dataset, Iterable)) for v in data.values()):
                return {k: self._get_len_recursive(v) for k, v in data.items()}
        elif isinstance(data, Sequence):
            data = list(data)
            if any(isinstance(v, (Mapping, Sequence, Dataset, Iterable)) for v in data):
                return [self._get_len_recursive(v) for v in data]

        length = sized_len(data)
        if length is None:
            raise ValueError(f"Couldn't compute the length of {data}")
        return length

    @functools.lru_cache(maxsize=1)
    def __len__(self) -> int:
        """Compute the length of `CombinedDataset` according to the `mode`."""
        all_lengths = self._get_len_recursive(self.datasets)
        mode = _supported_modes[self._mode]
        total_length = _reduce_data(all_lengths, mode["fn"], mode["default"])
        if isinstance(total_length, float):
            raise TypeError(f"The total size of the datasets must be an int, found {total_length}")
        return total_length


class CombinedLoader:
    """Combines different dataloaders and allows sampling in parallel.

    Args:
        loaders: the loaders to sample from. Can be all kind of collection
        mode:
            * ``"min_size"``, which raises StopIteration after the shortest loader (the one with the lowest number of
                batches) is done.
            * ``"max_size_cycle"`` which raises StopIteration after the longest loader (the one with most batches) is
                done, while cycling through the shorter loaders.
            * ``"sequential"`` which iterates through the laoaders in sequence.

    Examples:
        >>> loaders = {'a': DataLoader(range(6), batch_size=4),
        ...            'b': DataLoader(range(15), batch_size=5)}
        >>> combined_loader = CombinedLoader(loaders, 'max_size_cycle')
        >>> len(combined_loader)
        3
        >>> for item in combined_loader:
        ...     print(item)
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([10, 11, 12, 13, 14])}
        >>> combined_loader = CombinedLoader(loaders, 'min_size')
        >>> len(combined_loader)
        2
        >>> for item in combined_loader:
        ...     print(item)
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}
    """

    def __init__(self, loaders: Any, mode: str = "min_size"):
        if mode not in _supported_modes:
            raise ValueError(f"Unsupported mode {mode!r}, please select one of: {list(_supported_modes)}.")

        self.loaders = loaders

        datasets = apply_to_collection(
            self.loaders, Iterable, getattr, "dataset", None, wrong_dtype=(Sequence, Mapping)
        )
        # could be multiple datasets, but use self.dataset to follow the name convention in DataLoader
        self.dataset = CombinedDataset(datasets, mode)

        self._mode = mode
        self._wrap_loaders_max_size_cycle()

        self._iterator: Optional[Iterator] = None  # assigned in __iter__

    @property
    def sampler(self) -> Any:
        """Return a collections of samplers extracted from loaders."""
        return apply_to_collection(self.loaders, (DataLoader, IterableDataset), getattr, "sampler", None)

    @property
    def batch_sampler(self) -> Any:
        """Return a collections of batch samplers extracted from loaders."""
        return apply_to_collection(self.loaders, (DataLoader, IterableDataset), getattr, "batch_sampler", None)

    def _wrap_loaders_max_size_cycle(self) -> None:
        """Wraps all loaders to make sure they are cycled until the longest loader is exhausted.

        Returns:
            the wrapped loaders
        """
        if self._mode != "max_size_cycle" or not isinstance(self.loaders, (Sequence, Mapping)):
            return
        length = self._calc_num_batches()
        state = SharedCycleIteratorState()
        self.loaders = apply_to_collection(
            self.loaders, Iterable, CycleIterator, length=length, state=state, wrong_dtype=(Sequence, Mapping)
        )
        state.reset()

    def _apply_cycle_iterator_length(self) -> None:
        """When the model is `max_size_cycle`, compute the length across all ``CycleIterator`` and re-assign it to
        all dataloaders."""
        if self._mode != "max_size_cycle":
            return

        from lightning.pytorch.utilities.data import get_len

        def set_len(cycle_iterator: CycleIterator, length: int) -> None:
            cycle_iterator.length = length

        all_lengths = apply_to_collection(self.loaders, CycleIterator, lambda c: get_len(c.loader))
        max_length = _reduce_data(all_lengths, max, float("-inf"))
        apply_to_collection(self.loaders, CycleIterator, set_len, length=max_length)

    def __iter__(self) -> Any:
        """Create and return an iterator, `CombinedLoaderIterator`, for the combined loader."""

        # prevent `NotImplementedError` from PyTorch:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/dataloader.py#L541
        def __getstate__patch__(*_: Any) -> Dict:
            return {}

        _BaseDataLoaderIter.__getstate__ = __getstate__patch__  # type: ignore[assignment]
        iterator = CombinedLoaderIterator(self.loaders)
        self._iterator = iterator
        return iterator

    def _calc_num_batches(self) -> _NUMBER:
        from lightning.pytorch.utilities.data import get_len

        all_lengths = apply_to_collection(self.loaders, Iterable, get_len, wrong_dtype=(Sequence, Mapping))
        mode = _supported_modes[self._mode]
        return _reduce_data(all_lengths, mode["fn"], mode["default"])

    def __len__(self) -> int:
        """Compute the number of batches."""
        length = self._calc_num_batches()
        if isinstance(length, float):
            raise TypeError(f"Number of batches must be an int, found {length}")
        return length

    @staticmethod
    def _shutdown_workers_and_reset_iterator(dataloader: DataLoader) -> None:
        if hasattr(dataloader, "_iterator") and isinstance(dataloader._iterator, _MultiProcessingDataLoaderIter):
            dataloader._iterator._shutdown_workers()
        dataloader._iterator = None

    def reset(self) -> None:
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
        self._loader_iters: Any = None

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


def _reduce_data(data: Any, pairwise_reduction: Callable[[_NUMBER, _NUMBER], _NUMBER], default: _NUMBER) -> _NUMBER:
    if data is None:
        raise TypeError(f"Expected data to be int, Sequence or Mapping, but got {type(data).__name__}")

    total = default

    def reduce(v: _NUMBER) -> None:
        nonlocal total
        total = pairwise_reduction(total, v)

    apply_to_collection(data, (int, float), reduce)
    return total
