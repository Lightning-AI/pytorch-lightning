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
from typing import Any, Callable, Iterable, Iterator, List, Literal, Sized

from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, DataLoader
from typing_extensions import TypedDict

from lightning.fabric.utilities.data import sized_len
from lightning.pytorch.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten


class _CombinationMode(TypedDict):
    fn: Callable[[List[int]], int]


_supported_modes = {
    "min_size": _CombinationMode(fn=min),
    "max_size_cycle": _CombinationMode(fn=max),
}

_LITERAL_SUPPORTED_MODES = Literal["min_size", "max_size_cycle"]


class _CombinedDataset(Sized):
    """Combine multiple datasets."""

    def __init__(self, datasets: Any, mode: _LITERAL_SUPPORTED_MODES = "min_size"):
        """
        Args:
            datasets: Collections of Iterables.
            mode: Mode to use when computing the length.
        """
        if mode not in _supported_modes:
            raise ValueError(f"Unsupported mode {mode!r}, please select one of: {list(_supported_modes)}.")
        self._mode = mode
        self._datasets = datasets

    @property
    def datasets(self) -> Any:
        return self._datasets

    def __len__(self) -> int:
        """Compute the length of `CombinedDataset` according to the `mode`."""
        lengths = [length for ds in _tree_flatten(self._datasets)[0] if (length := sized_len(ds)) is not None]
        if not lengths:
            raise NotImplementedError("All datasets are iterable-style datasets.")
        fn = _supported_modes[self._mode]["fn"]
        return fn(lengths)


class CombinedLoader(Iterable):
    """Combines different dataloaders and allows sampling in parallel.

    Args:
        loaders: the loaders to sample from. Can be all kind of collection
        mode:
            * ``"min_size"``, which raises StopIteration after the shortest loader (the one with the lowest number of
                batches) is done.
            * ``"max_size_cycle"`` which raises StopIteration after the longest loader (the one with most batches) is
                done, while cycling through the shorter loaders.

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

    def __init__(self, loaders: Any, mode: _LITERAL_SUPPORTED_MODES = "min_size") -> None:
        if mode not in _supported_modes:
            raise ValueError(f"Unsupported mode {mode!r}, please select one of: {list(_supported_modes)}.")
        self._loaders = loaders
        self._loaders_flattened, self._loaders_spec = _tree_flatten(loaders)

        # TODO(carlos): doing this might not be necessary
        datasets = _map_and_unflatten(
            lambda x: getattr(x, "dataset", None), self._loaders_flattened, self._loaders_spec
        )
        # could be multiple datasets, but use self.dataset to follow the name convention in DataLoader
        self.dataset = _CombinedDataset(datasets, mode)

        self._mode = mode
        self._loader_iters: List[Iterator] = []
        self._consumed: List[bool] = []

    @property
    def loaders(self) -> Any:
        """Return the original collection of loaders."""
        return self._loaders

    @property
    def sampler(self) -> Any:
        """Return a collections of samplers extracted from loaders."""
        return _map_and_unflatten(lambda x: getattr(x, "sampler", None), self._loaders_flattened, self._loaders_spec)

    @property
    def batch_sampler(self) -> Any:
        """Return a collections of batch samplers extracted from loaders."""
        return _map_and_unflatten(
            lambda x: getattr(x, "batch_sampler", None), self._loaders_flattened, self._loaders_spec
        )

    def __next__(self) -> Any:
        n = len(self._loader_iters)
        out = [None] * n  # values per iterator
        for i in range(n):
            try:
                out[i] = next(self._loader_iters[i])
            except StopIteration:
                self._consumed[i] = True
                if all(self._consumed):
                    raise
                if self._mode == "max_size_cycle":
                    # reset the consumed dataloader
                    self._loader_iters[i] = iter(self._loaders_flattened[i])
                    out[i] = next(self._loader_iters[i])
                    continue
                elif self._mode == "min_size":
                    raise
        return tree_unflatten(out, self._loaders_spec)

    def __iter__(self) -> Iterator:
        self._loader_iters = [iter(loader) for loader in self._loaders_flattened]
        self._consumed = [False] * len(self._loaders_flattened)
        return self

    def __len__(self) -> int:
        """Compute the number of batches."""
        lengths = []
        for dl in self._loaders_flattened:
            length = sized_len(dl)
            if length is None:
                raise NotImplementedError(f"`{type(dl).__name__}` does not define `__len__`")
            lengths.append(length)
        fn = _supported_modes[self._mode]["fn"]
        return fn(lengths)

    def reset(self) -> None:
        self._loader_iters = []
        self._consumed = []
        for loader in self._loaders_flattened:
            _shutdown_workers_and_reset_iterator(loader)

    def _update_index(self, dataloader: Iterable, index: int) -> None:
        # mutation needs to be done using this method to avoid stale references
        self._loaders_flattened[index] = dataloader
        self._loaders = tree_unflatten(self._loaders_flattened, self._loaders_spec)


def _shutdown_workers_and_reset_iterator(dataloader: DataLoader) -> None:
    if hasattr(dataloader, "_iterator"):
        if isinstance(dataloader._iterator, _MultiProcessingDataLoaderIter):
            dataloader._iterator._shutdown_workers()
        dataloader._iterator = None
