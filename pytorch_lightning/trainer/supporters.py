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

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter, DataLoader
from torch.utils.data.dataset import IterableDataset

from pytorch_lightning.utilities.apply_func import apply_to_collection, apply_to_collections
from pytorch_lightning.utilities.auto_restart import (
    _cycle_to_next_worker_and_reset,
    _find_current_worker,
    CaptureIterableDataset,
)
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.data import get_len
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_enabled


class TensorRunningAccum:
    """Tracks a running accumulation values (min, max, mean) without graph
    references.

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

    def reset(self) -> None:
        """Empty the accumulator."""
        self.__init__(self.window_length)

    def last(self):
        """Get the last added element."""
        if self.last_idx is not None:
            return self.memory[self.last_idx]

    def append(self, x):
        """Add an element to the accumulator."""
        if self.memory is None:
            self.memory = torch.zeros(self.window_length, *x.shape)

        # ensure same device and type
        if self.memory.device != x.device or self.memory.type() != x.type():
            x = x.to(self.memory)

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
                return getattr(self.memory, how)()
            return getattr(self.memory[: self.current_idx], how)()


class PredictionCollection:
    def __init__(self, global_rank: int, world_size: int):
        self.global_rank = global_rank
        self.world_size = world_size
        self.predictions = {}
        self.num_predictions = 0

    def _add_prediction(self, name, values, filename):
        if filename not in self.predictions:
            self.predictions[filename] = {name: values}
        elif name not in self.predictions[filename]:
            self.predictions[filename][name] = values
        elif isinstance(values, Tensor):
            self.predictions[filename][name] = torch.cat((self.predictions[filename][name], values))
        elif isinstance(values, list):
            self.predictions[filename][name].extend(values)

    def add(self, predictions):

        if predictions is None:
            return

        for filename, pred_dict in predictions.items():
            for feature_name, values in pred_dict.items():
                self._add_prediction(feature_name, values, filename)

    def to_disk(self) -> None:
        """Write predictions to file(s)."""
        for filepath, predictions in self.predictions.items():
            fs = get_filesystem(filepath)
            # normalize local filepaths only
            if fs.protocol == "file":
                filepath = os.path.realpath(filepath)
            if self.world_size > 1:
                stem, extension = os.path.splitext(filepath)
                filepath = f"{stem}_rank_{self.global_rank}{extension}"
            dirpath = os.path.split(filepath)[0]
            fs.mkdirs(dirpath, exist_ok=True)

            # Convert any tensor values to list
            predictions = {k: v if not isinstance(v, Tensor) else v.tolist() for k, v in predictions.items()}

            # Check if all features for this file add up to same length
            feature_lens = {k: len(v) for k, v in predictions.items()}
            if len(set(feature_lens.values())) != 1:
                raise ValueError("Mismatching feature column lengths found in stored EvalResult predictions.")

            # Switch predictions so each entry has its own dict
            outputs = []
            for values in zip(*predictions.values()):
                output_element = dict(zip(predictions.keys(), values))
                outputs.append(output_element)

            # Write predictions for current file to disk
            with fs.open(filepath, "wb") as fp:
                torch.save(outputs, fp)


@dataclass
class SharedCycleIteratorState:

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
            raise MisconfigurationException("Please, call reset once all dataloaders have been added.")
        if len(self.dataloaders) == 1:
            return False
        decision_fn = all if self.mode == "max_size_cycle" else any
        return decision_fn(self.has_finished.values())


class CycleIterator:
    """
    Iterator for restarting a dataloader if it runs out of samples
    """

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

    def __iter__(self) -> Any:
        """
        Creates the internal iterator and returns self

        Returns:
            CycleIterator: self
        """
        self.counter = 0
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
            return next(self._loader_iter)

        finally:
            self.counter += 1

    def __len__(self) -> Union[int, float]:
        return self.length


class CombinedDataset:
    """
    Combine multiple datasets and compute their statistics
    """

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
        """
        Compute the length of `CombinedDataset` according to the `mode`.

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


class DataLoaderDict(Dict):
    # behaves exactly like a dict, this is used to simplify apply_to_collection.
    pass


class CombinedLoader:
    """
    Combines different dataloaders and allows sampling in parallel.
    Supported modes are 'min_size', which raises StopIteration after the shortest loader
    (the one with the lowest number of batches) is done, and 'max_size_cycle` which raises
    StopIteration after the longest loader (the one with most batches) is done, while cycling
    through the shorter loaders.

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
    def _state_dict_fn(dataloader: DataLoader, iterator: Optional[Iterator], num_batches_processed: int) -> Dict:
        # find next worker if multiple workers were used
        state = _find_current_worker(iterator)
        if isinstance(dataloader.dataset, CaptureIterableDataset):
            # the sampler state dict are extracted in `CombinedLoaderIterator`
            if iterator is not None and getattr(iterator, "_sampler_state_dict", None) is not None:
                state.update(iterator._sampler_state_dict[0])
        else:
            # fetch directly from fast forward sampler
            state.update(dataloader.fast_forward_sampler.state_dict(num_batches_processed))
        return DataLoaderDict(state)

    def state_dict(self, num_batches_processed: int) -> Dict:
        """
        The state dict includes all states from wrapped dataloaders and their samplers through the
        ``CaptureIterableDataset`` and fast-forward samplers.

        Args:
            num_batches_processed: The number of batches processed so far, needed because the individual dataloaders
                may have already prefetched more batches by the time a state dict is requested.
        """
        if not _fault_tolerant_enabled():
            return DataLoaderDict()

        state_dict_fn = partial(self._state_dict_fn, num_batches_processed=num_batches_processed)

        return apply_to_collections(self.loaders, self._iterator.loader_iters, (Iterator, DataLoader), state_dict_fn)

    def load_state_dict(self, state_dict):
        # store the samplers state.
        # They would be reloaded once the `CombinedIterator` as been created
        # and the workers are created.
        self._loaders_iter_state_dict = state_dict

        def mock_reset_fn(self, *_, **__):
            pass

        # mock reset call, so we can rotate the `_worker_queue_idx_cycle` to failed worker
        # and get the first batch from it
        _MultiProcessingDataLoaderIter._original_reset = _MultiProcessingDataLoaderIter._reset
        _MultiProcessingDataLoaderIter._reset = mock_reset_fn

    def on_restart(self, iterator: Iterator):
        if not self._loaders_iter_state_dict:
            return

        # this happen inside the workers if any were specificied.

        def create_loader_iters(dataloader: DataLoader, state_dict: DataLoaderDict):
            if isinstance(dataloader.dataset, CaptureIterableDataset):
                # provide the `state_dict` to the `CaptureIterableDataset`
                # as it is responsible for passing down the state to associated `FastForwardSampler`
                dataloader.dataset.load_state_dict(state_dict)
            else:
                # for `Mapping-based` dataset, the `fast_forward_sampler` was attached
                # on the dataloader for simplicity
                dataloader.fast_forward_sampler.load_state_dict(state_dict)

            # cycle back the iterator to the failed worker if multiple workers were provided
            iterator = _cycle_to_next_worker_and_reset(dataloader, state_dict)

            if isinstance(dataloader.dataset, CaptureIterableDataset):
                # remove keys related to iterator
                state_dict = {k: v for k, v in state_dict.items() if k not in ("num_worker", "previous_worker")}
                # need to re-attach the state dict into the iterator for future collection.
                iterator._sampler_state_dict = [state_dict]
            return iterator

        # apply the `create_loader_iters` on the collection of `DataLoader / Iterator`.
        # each `Iterator` was created from the `DataLoader`.
        iterator._loader_iters = apply_to_collections(
            self.loaders, self._loaders_iter_state_dict, (DataLoader, DataLoaderDict), create_loader_iters
        )

    @property
    def sampler(self) -> Union[Iterable, Sequence, Mapping]:
        """Return a collections of samplers extracting from loaders."""
        return apply_to_collection(self.loaders, (DataLoader, IterableDataset), getattr, "sampler", None)

    def _wrap_loaders_max_size_cycle(self) -> Any:
        """
        Wraps all loaders to make sure they are cycled until the longest loader is exhausted

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

    def __iter__(self) -> Any:
        """
        Create and return an iterator, `CombinedLoaderIterator`, for the combined loader.
        """

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
    def _calc_num_batches(loaders: Any) -> Union[int, float]:
        """
        Compute the length (aka the number of batches) of `CombinedLoader`.

        Args:
            loaders: a collections of loaders.

        Returns:
            length: the minimum length of loaders
        """
        all_lengths = apply_to_collection(loaders, Iterable, get_len, wrong_dtype=(Sequence, Mapping))

        if isinstance(all_lengths, (int, float)):
            return all_lengths
        return _nested_calc_num_data(all_lengths, min)

    def __len__(self) -> int:
        return self._calc_num_batches(self.loaders)


class CombinedLoaderIterator:
    """
    Custom Iterator returning data from multple loaders, and allows sampling in parallel
    """

    def __init__(self, loaders: Any):
        """
        Args:
            loaders: the loaders to sample from. Can be all kind of collection
        """
        self.loaders = loaders
        self._loader_iters = None

    @property
    def loader_iters(self) -> Any:
        """
        Get the `_loader_iters` and create one if it is None.
        """
        if self._loader_iters is None:
            self._loader_iters = self.create_loader_iters(self.loaders)

        return self._loader_iters

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        """
        Fetches the next batch from multiple data loaders

        Returns:
            a collections of batch data
        """
        return self.request_next_batch(self.loader_iters)

    @staticmethod
    def request_next_batch(loader_iters: Union[Iterator, Sequence, Mapping]) -> Any:
        """
        Return the batch of data from multiple iterators.

        Args:
            loader_iters: a collections of iterators

        Returns
            Any: a collections of batch data
        """

        def next_fn(iterator: Iterator):
            batch = next(iterator)
            if not _fault_tolerant_enabled():
                return batch
            # when fault tolerant is enabled, the iterator will return
            # `FastForwardSampler` state_dict metadata
            # along side with the user data.
            # the metadata are extracted and store directly on the iterator
            # to simplify the collection on `state_dict` call.
            batch, samplers_state_dict = CaptureIterableDataset.extract_samplers_state_dict_from_batch(batch)
            # store the `sampler_state_dict` on the iterator
            CaptureIterableDataset.store_samplers_state_dict(iterator, samplers_state_dict)
            return batch

        return apply_to_collection(loader_iters, Iterator, next_fn)

    @staticmethod
    def create_loader_iters(
        loaders: Union[Any, Iterator, Sequence, Mapping]
    ) -> Union[Any, Iterator, Sequence, Mapping]:
        """
        Create and return a collection of iterators from loaders.

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


def prefetch_iterator(iterable: Iterable) -> Generator[Tuple[Any, bool], None, None]:
    """
    Returns an iterator that pre-fetches and caches the next item.
    The values are passed through from the given iterable with an added boolean indicating if this is the last item.
    See `https://stackoverflow.com/a/1630350 <https://stackoverflow.com/a/1630350>`_
    """
    it = iter(iterable)

    try:
        # the iterator may be empty from the beginning
        last = next(it)
    except StopIteration:
        return

    for val in it:
        # yield last and has next
        yield last, False
        last = val
    # yield last, no longer has next
    yield last, True
