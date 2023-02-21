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
from collections.abc import Iterable
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple, Type, TypeVar, Union

from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict

from lightning.fabric.utilities.data import sized_len
from lightning.pytorch.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten

_T = TypeVar("_T")


class _ModeIterator(Iterator[_T]):
    def __init__(self, iterables: List[Iterable]) -> None:
        self.iterables = iterables
        self.iterators: List[Iterator] = []

    def __next__(self) -> _T:
        raise NotImplementedError

    def __iter__(self) -> Self:  # type: ignore[valid-type]
        self.iterators = [iter(iterable) for iterable in self.iterables]
        return self

    def reset(self) -> None:
        self.iterators = []


class _MaxSizeCycle(_ModeIterator[List]):
    def __init__(self, iterables: List[Iterable]) -> None:
        super().__init__(iterables)
        self._consumed: List[bool] = []

    def __next__(self) -> List:
        n = len(self.iterators)
        out = [None] * n  # values per iterator
        for i in range(n):
            try:
                out[i] = next(self.iterators[i])
            except StopIteration:
                self._consumed[i] = True
                if all(self._consumed):
                    raise
                # reset the consumed dataloader
                self.iterators[i] = iter(self.iterables[i])
                out[i] = next(self.iterators[i])
        return out

    def __iter__(self) -> Self:  # type: ignore[valid-type]
        super().__iter__()
        self._consumed = [False] * len(self.iterables)
        return self

    def reset(self) -> None:
        super().reset()
        self._consumed = []


class _MinSize(_ModeIterator[List]):
    def __next__(self) -> List:
        return [next(it) for it in self.iterators]


class _Sequential(_ModeIterator[Tuple[Any, int, int]]):
    def __init__(self, iterables: List[Iterable], limits: Optional[List[Union[int, float]]] = None) -> None:
        super().__init__(iterables)
        self._iterator_idx = 0  # what would be dataloader_idx
        self._idx = 0  # what would be batch_idx
        self.limits = limits

    @property
    def limits(self) -> Optional[List[Union[int, float]]]:
        """Optional limits per iterator."""
        return self._limits

    @limits.setter
    def limits(self, limits: Optional[List[Union[int, float]]]) -> None:
        if limits is not None and len(limits) != len(self.iterables):
            raise ValueError(
                f"Mismatch in number of limits ({len(limits)}) and number of iterables ({len(self.iterables)})"
            )
        self._limits = limits

    def __next__(self) -> Tuple[Any, int, int]:
        n = len(self.iterators)
        if n == 0 or self._iterator_idx >= n:
            raise StopIteration

        # if limits are set, go to the correct iterator
        if self.limits is not None:
            while self.limits[self._iterator_idx] <= self._idx:
                self._use_next_iterator()
                if self._iterator_idx >= n:
                    raise StopIteration

        try:
            out = next(self.iterators[self._iterator_idx])
            index = self._idx
            self._idx += 1
            # batch, batch_idx, dataloader_idx
            return out, index, self._iterator_idx
        except StopIteration:
            # try the next iterator
            self._use_next_iterator()
            return self.__next__()

    def __iter__(self) -> Self:  # type: ignore[valid-type]
        super().__iter__()
        self._iterator_idx = 0
        self._idx = 0
        return self

    def reset(self) -> None:
        super().reset()
        self._iterator_idx = 0
        self._idx = 0

    def _use_next_iterator(self) -> None:
        self._iterator_idx += 1
        self._idx = 0


class _CombinationMode(TypedDict):
    fn: Callable[[List[int]], int]
    iterator: Type[_ModeIterator]


_supported_modes = {
    "min_size": _CombinationMode(fn=min, iterator=_MinSize),
    "max_size_cycle": _CombinationMode(fn=max, iterator=_MaxSizeCycle),
    "sequential": _CombinationMode(fn=sum, iterator=_Sequential),
}

_LITERAL_SUPPORTED_MODES = Literal["min_size", "max_size_cycle", "sequential"]


class CombinedLoader(Iterable):
    """Combines different iterables under custom sampling modes.

    Args:
        iterables: the loaders to sample from. Can be any kind of collection
        mode:
            * ``"min_size"``, which raises StopIteration after the shortest iterable (the one with the lowest number of
                items) is done.
            * ``"max_size_cycle"`` which raises StopIteration after the longest iterable (the one with most items) is
                done, while cycling through rest of the iterables.
            * ``"sequential"`` will consume ecah iterable sequentially, and returns a tuple with the associated index
                from each iterable.

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> iterables = {'a': DataLoader(range(6), batch_size=4),
        ...              'b': DataLoader(range(15), batch_size=5)}
        >>> combined_loader = CombinedLoader(iterables, 'max_size_cycle')
        >>> len(combined_loader)
        3
        >>> for batch in combined_loader:
        ...     print(batch)
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([10, 11, 12, 13, 14])}
        >>> combined_loader = CombinedLoader(iterables, 'min_size')
        >>> len(combined_loader)
        2
        >>> for batch in combined_loader:
        ...     print(batch)
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}
        >>> combined_loader = CombinedLoader(iterables, 'sequential')
        >>> len(combined_loader)
        5
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch} {batch_idx=} {dataloader_idx=}")
        tensor([0, 1, 2, 3]) batch_idx=0 dataloader_idx=0
        tensor([4, 5]) batch_idx=1 dataloader_idx=0
        tensor([0, 1, 2, 3, 4]) batch_idx=0 dataloader_idx=1
        tensor([5, 6, 7, 8, 9]) batch_idx=1 dataloader_idx=1
        tensor([10, 11, 12, 13, 14]) batch_idx=2 dataloader_idx=1
    """

    def __init__(self, iterables: Any, mode: _LITERAL_SUPPORTED_MODES = "min_size") -> None:
        if mode not in _supported_modes:
            raise ValueError(f"Unsupported mode {mode!r}, please select one of: {list(_supported_modes)}.")
        self._iterables = iterables
        self._flattened, self._spec = _tree_flatten(iterables)
        self._mode = mode
        self._iterator: Optional[_ModeIterator] = None

    @property
    def iterables(self) -> Any:
        """Return the original collection of iterables."""
        return self._iterables

    @property
    def sampler(self) -> Any:
        """Return a collections of samplers extracted from iterables."""
        return _map_and_unflatten(lambda x: getattr(x, "sampler", None), self.flattened, self._spec)

    @property
    def batch_sampler(self) -> Any:
        """Return a collections of batch samplers extracted from iterables."""
        return _map_and_unflatten(lambda x: getattr(x, "batch_sampler", None), self.flattened, self._spec)

    @property
    def flattened(self) -> List[Any]:
        """Return the flat list of iterables."""
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: List[Any]) -> None:
        if len(flattened) != len(self._flattened):
            raise ValueError(
                f"Mismatch in flattened length ({len(flattened)}) and existing length ({len(self._flattened)})"
            )
        # update the iterable collection
        self._iterables = tree_unflatten(flattened, self._spec)
        self._flattened = flattened

    def __next__(self) -> Any:
        assert self._iterator is not None
        out = next(self._iterator)
        if isinstance(self._iterator, _Sequential):
            return out
        return tree_unflatten(out, self._spec)

    def __iter__(self) -> Self:  # type: ignore[valid-type]
        cls = _supported_modes[self._mode]["iterator"]
        iterator = cls(self.flattened)
        iter(iterator)
        self._iterator = iterator
        return self

    def __len__(self) -> int:
        """Compute the number of batches."""
        lengths = []
        for dl in self.flattened:
            length = sized_len(dl)
            if length is None:
                raise NotImplementedError(f"`{type(dl).__name__}` does not define `__len__`")
            lengths.append(length)
        fn = _supported_modes[self._mode]["fn"]
        return fn(lengths)

    def reset(self) -> None:
        if self._iterator is not None:
            self._iterator.reset()
            self._iterator = None
        for iterable in self.flattened:
            _shutdown_workers_and_reset_iterator(iterable)

    def _dataset_length(self) -> int:
        """Compute the total length of the datasets according to the `mode`."""
        datasets = [getattr(dl, "dataset", None) for dl in self.flattened]
        lengths = [length for ds in datasets if (length := sized_len(ds)) is not None]
        if not lengths:
            raise NotImplementedError("All datasets are iterable-style datasets.")
        fn = _supported_modes[self._mode]["fn"]
        return fn(lengths)


def _shutdown_workers_and_reset_iterator(dataloader: object) -> None:
    if hasattr(dataloader, "_iterator"):
        if isinstance(dataloader._iterator, _MultiProcessingDataLoaderIter):
            dataloader._iterator._shutdown_workers()
        dataloader._iterator = None
