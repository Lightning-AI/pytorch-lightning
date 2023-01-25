# Copyright The Lightning team.
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
from collections.abc import Sized
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union

from lightning_utilities.core.apply_func import apply_to_collection
from torch.utils.data import Dataset, DistributedSampler, get_worker_info, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
    DataLoader,
    IterableDataset,
)
from typing_extensions import TypedDict

import pytorch_lightning as pl
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities.distributed import _collect_states_on_rank_zero
from pytorch_lightning.utilities.enums import _FaultTolerantMode, AutoRestartBatchKeys
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import _collect_rng_states, _set_rng_states


class _IteratorStateDict(TypedDict):
    dataset_state: Dict[int, Any]
    sampler_state: Dict[int, Any]
    worker_id: int
    num_workers: int
    num_batches_fetched: int
    name: Optional[str]


class _MergedIteratorStateDict(TypedDict):
    state: Dict[str, Any]
    latest_worker_id: int
    represent_map_dataset: Optional[bool]


class FastForwardSampler(Sampler):
    """This FastForwardSampler wraps a :class:`torch.utils.data.Sampler` and records the number of iterations
    performed during an epoch.

    It maintains a state, saved with :meth:`state_dict`, that can be reloaded with
    :meth:`load_state_dict`. If the sampler is used in a multiprocessing context, the ``FastForwardSampler`` will record
    the state of the current worker.
    When reloading, the ``FastForwardSampler`` will "fast-forward" the wrapped sampler by iterating through all the
    samples seen in the last iterations (for the current worker).
    """

    def __init__(self, sampler: Union[Sampler, Iterable], attr_name: Optional[str] = None) -> None:
        super().__init__(data_source=None)
        self._sampler = sampler
        self.restarting: bool = False
        self._current_iteration = 0
        self._counter = 0
        self._dataloader_batch_size: Optional[int] = None
        self._cached_state_dict: Optional[Dict[int, Any]] = None
        self._attr_name = attr_name

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        return getattr(self._sampler, key, None)

    def setup(self, dataloader_batch_size: Optional[int] = None) -> None:
        """Setup the ``FastForwardSampler``.

        This is required only when the provided dataset subclassed
        :class:`torch.utils.data.Dataset`.
        """
        self._dataloader_batch_size = dataloader_batch_size

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    def __iter__(self) -> Iterator[Any]:
        self.sampler_iter = iter(self._sampler)
        self._current_iteration = 0
        self._counter = 0
        return self

    def __next__(self) -> Any:
        # the `state dict` was cached as workers were unavailable before.
        if self._cached_state_dict is not None:
            self._load_non_random_state(self._cached_state_dict)

        while self._counter < self._current_iteration:
            next(self.sampler_iter)
            self._counter += 1

        # here: i == self._current_iteration
        if self._cached_state_dict is not None:
            self._cached_state_dict = None

        # recreate iterator to be sure loading is reflected there as well
        self._current_iteration += 1
        self._counter += 1
        has_raised = False
        try:
            return next(self.sampler_iter)
        except StopIteration:
            has_raised = True

        self._current_iteration = 0
        self._counter = 0
        self._cached_state_dict = None
        self.restarting = False
        if has_raised:
            raise StopIteration

    def __len__(self) -> int:
        assert isinstance(self._sampler, Sized)
        return len(self._sampler)

    def state_dict(self, num_batches_processed: Optional[int] = None) -> Dict[int, Dict[str, int]]:
        """Returns the state of the sampler in the current worker.

        The worker id indexes the state dict.
        """
        return {self.worker_id: {"current_iteration": self._compute_current_iteration(num_batches_processed)}}

    def load_state_dict(self, state_dict: Dict[int, Any]) -> None:
        """Loads the saved state for the wrapped sampler.

        If the ``state_dict`` contains multiple states, it means there were multiple workers. The state will be cached
        and fully reloaded (fast-forward) the first time :meth:`__iter__` is called.
        """
        # as workers aren't available, the ``state_dict``` is cached until workers are made available.
        state_dict = deepcopy(state_dict)
        self._cached_state_dict = state_dict
        self.restarting = True

    def _compute_current_iteration(self, num_batches_processed: Optional[int] = None) -> int:
        """This function is used to compute the effective iteration.

        As DataLoader can perform ``prefecthing`` or training can fail while processing a batch, the current iteration
        needs to be computed using the ``num_batches_processed`` processed information.
        """
        if num_batches_processed is not None:
            current_iteration = num_batches_processed
        else:
            current_iteration = self._current_iteration

        if self._dataloader_batch_size and num_batches_processed is not None:
            current_iteration *= self._dataloader_batch_size

        return current_iteration

    def _load_non_random_state(self, state_dict: Dict[int, Dict[str, Any]]) -> None:
        self._current_iteration = state_dict[self.worker_id]["current_iteration"]


@dataclass(frozen=True, unsafe_hash=True)
class IteratorState:
    """The state of an iterator in a single worker process."""

    dataset_state: Dict[int, Any] = field(default_factory=dict)
    sampler_state: Dict[int, Any] = field(default_factory=dict)
    worker_id: int = 0
    num_workers: int = 0
    num_batches_fetched: int = 0
    name: Optional[str] = None

    @classmethod
    def from_state_dict(cls, state_dict: _IteratorStateDict) -> "IteratorState":
        return cls(**state_dict)


@dataclass
class MergedIteratorState:
    """This class is used to hold the current iterator state and lives on the iterator.

    It holds the current merged states from all worker processes. Once an iterator advances, it can store updates of the
    worker states in this merged iterator state.
    """

    state: Dict = field(default_factory=dict)
    latest_worker_id: int = 0
    represent_map_dataset: Optional[bool] = None

    def update(self, generator_name: Optional[str], new_state: IteratorState) -> None:
        # a map based dataset doesn't own a generator and therefore `generator_name` should be None.
        self.represent_map_dataset = generator_name is None
        latest_worker_id = new_state.worker_id
        if generator_name is None:
            self.state[latest_worker_id] = new_state
        else:
            if generator_name not in self.state:
                self.state[generator_name] = {}
            state = self.state[generator_name]
            state[latest_worker_id] = new_state

        self.latest_worker_id = latest_worker_id

    @property
    def sampler_states(self) -> Dict[int, Any]:
        """Returns the merged sampler states for all worker processes."""
        return {0: self.state[k].sampler_state[0] for k in self.state.keys()}

    @property
    def dataset_states(self) -> Dict[int, Any]:
        """Returns the merged dataset states for all worker processes."""
        return {k: self.state[k].dataset_state[k] for k in self.state.keys()}

    @classmethod
    def from_state_dict(cls, state_dict: _MergedIteratorStateDict) -> "MergedIteratorState":
        if state_dict["represent_map_dataset"]:
            state_dict["state"] = {
                worker_id: IteratorState.from_state_dict(state) for worker_id, state in state_dict["state"].items()
            }
        else:
            state_dict["state"] = {
                sampler_name: {
                    worker_id: IteratorState.from_state_dict(state) for worker_id, state in worker_state.items()
                }
                for sampler_name, worker_state in state_dict["state"].items()
            }
        return cls(**state_dict)

    def __len__(self) -> int:
        return len(self.state)


class CaptureMapDataset(Dataset):
    """This class is used to capture the state from the map-based state dataset.

    Note:
        We currently don't support restoring if we fail during the first `N = num_workers` batches, where
        `num_workers` is the number of workers spawned by the dataloader.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset: Dataset = dataset
        self._cached_state_dict: Optional[Dict[int, Any]] = None

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    def __getitem__(self, item: int) -> Tuple[Any, Dict[int, Dict]]:
        if self._cached_state_dict is not None:
            if self.worker_id in self._cached_state_dict:
                _set_rng_states(self._cached_state_dict[self.worker_id]["rng_states"])
            self._cached_state_dict = None

        return self.dataset[item]

    def __len__(self) -> int:
        assert isinstance(self.dataset, Sized)
        return len(self.dataset)

    def load_state_dict(self, state_dict: Dict[int, Any], latest_worker_id: int, num_workers: int) -> None:
        # as workers aren't available, the ``state_dict``` is cached until workers are made available.
        self._cached_state_dict = _rotate_worker_indices(deepcopy(state_dict), latest_worker_id, num_workers)

    def state_dict(self) -> Dict[int, Dict[str, Any]]:
        return {self.worker_id: {"rng_states": _collect_rng_states()}}


class CaptureIterableDataset(IterableDataset):
    """The ``CaptureIterableDataset`` is used to wrap an :class:`torch.utils.data.IterableDataset`.

    On ``__iter__`` function call,   the ``CaptureIterableDataset`` will wrap the wrapped dataset     generators into
    ``FastForwardSampler`` to keep track of progress. On ``__next__`` function call, the ``CaptureIterableDataset`` will
    return a dictionary containing     user data and metadata containing the ``FastForwardSampler`` samplers state_dict.
    """

    def __init__(self, dataset: IterableDataset) -> None:
        super().__init__()
        self.dataset = deepcopy(dataset)
        self.samplers: Optional[Dict[str, FastForwardSampler]] = None
        self._state_dict: Optional[Dict[str, Any]] = None
        self._has_wrapped: bool = False

    @property
    def sampler(self) -> Sampler:
        return self.dataset.sampler

    def state_dict(self) -> Dict[str, Any]:
        assert self.samplers is not None
        return {k: v.state_dict() for k, v in self.samplers.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._state_dict = deepcopy(state_dict)

    def _wrap_generator_samplers(self) -> None:
        self.samplers = {}

        # access wrapped dataset attributes
        dataset_dict = self.dataset.__dict__

        # create a dictionary of generator present within the dataset attributes
        dataset_sampler_generators = {k: v for k, v in dataset_dict.items() if isinstance(v, (Generator, Iterator))}

        # iterate over the generator. If a generator was created from a `Sampler`,
        # it will be wrapped into a `FastForwardSampler`.
        for (generator_attr_name, generator) in dataset_sampler_generators.items():

            if isinstance(generator, Sampler):
                continue

            # wrap the generator into a `FastForwardSampler`
            sampler = FastForwardSampler(generator, attr_name=generator_attr_name)

            # if `CaptureIterableDataset` was available, the sampler should reload its own state.
            if self._state_dict is not None:
                sampler.load_state_dict(self._state_dict[generator_attr_name])
            # store the samplers
            self.samplers[generator_attr_name] = sampler

            # replace generator with the generator from the `FastForwardSampler`.
            dataset_dict[generator_attr_name] = iter(sampler)

        self.reset_on_epoch()

    def reset_on_epoch(self) -> None:
        self._state_dict = None

    def __iter__(self) -> Iterator:
        # create a generator from the wrapped Iterative Dataset
        # if the dataset contained samplers, they will be transformed into generators
        self.iter_data = iter(self.dataset)

        # wrap any generator associated to a Sampler into a `FastForwardSampler`.
        if isinstance(self.iter_data, Generator):
            raise MisconfigurationException(
                "PyTorch Lightning Fault-Tolerant feature does not support `__iter__` returning a generator."
                " Please use the `__next__` function to fetch the next batch and use a sampler for"
                " doing your iterations."
            )
        self._wrap_generator_samplers()
        return self

    def __next__(self) -> Dict[str, Any]:
        return next(self.iter_data)


def _find_fast_forward_samplers(dataloader: DataLoader) -> Optional[FastForwardSampler]:
    """If the ``DataLoader`` is wrapping a mapping based Dataset, return the ``FastForwardSampler``."""
    if isinstance(dataloader.sampler, FastForwardSampler):
        return dataloader.sampler

    if isinstance(dataloader.batch_sampler, FastForwardSampler):
        return dataloader.batch_sampler


def _cycle_to_next_worker_and_reset(dataloader: DataLoader, state_dict: Dict[str, Any]) -> Iterator:
    """This function is used to cycle back the DataLoader ``_MultiProcessingDataLoaderIter`` workers and call the
    reset function.

    Returns:
        iterator: Return the iterator generated from the provided ``DataLoader``.
    """
    # create iterator from dataloader
    iter_dataloader = iter(dataloader)
    # get current num workers
    num_workers = getattr(iter_dataloader, "_num_workers", 0)
    # as `state_dict` are workers dependent, Lightning doesn't support changing
    # the `num_workers` for Fault-tolerance
    if state_dict["num_workers"] != num_workers:
        raise MisconfigurationException(
            f"The provided `num_workers` {num_workers} doesn't match the one used "
            f"while generating the checkpoint: {state_dict['num_workers']}"
        )
    # when using multiple workers, we will cycle back the worker queue idx to
    # start back on the failed worker.
    if isinstance(iter_dataloader, _MultiProcessingDataLoaderIter):
        # move back to 0
        while next(iter_dataloader._worker_queue_idx_cycle) != 0:
            pass
        # increment previous worker
        if isinstance(state_dict["previous_worker"], int):
            for _ in range(state_dict["previous_worker"] - 1):
                next(iter_dataloader._worker_queue_idx_cycle)

        # we can finally call reset and apply prefetching.
        iter_dataloader._reset = iter_dataloader._original_reset  # type: ignore[assignment]
        iter_dataloader._reset(dataloader, first_iter=True)
    # return the iterator
    return iter_dataloader


def _capture_metadata_collate(
    samples: List, dataset: Dataset, collate_fn: Callable, fault_tolerant_mode: _FaultTolerantMode
) -> Any:
    """A collate_fn function that adds the state dict of a :class:`CaptureIterableDataset` or
    :class:`CaptureMapDataset` used in the worker processes. This function gets executed within the worker
    processes. The structure will be:

    .. code-block:: python

        {
            "data": ...,  # data returned by Dataset
            "__pl_restart_meta": {"sampler_name0": state_dict0, "sampler_name1": state_dict1},
        }
    """
    data = collate_fn(samples)
    metadata = None
    if fault_tolerant_mode.is_automatic:
        metadata = dataset.state_dict()
    else:
        state_dict_fn = getattr(dataset, "state_dict", None)
        info = get_worker_info()
        worker_id = info.id if info else 0
        if state_dict_fn is not None:
            metadata = state_dict_fn()
            if worker_id not in metadata:
                if info and info.num_workers > 1:
                    raise MisconfigurationException(
                        f"The state_dict returned by {dataset} needs to be indexed by `worker_id` integer keys."
                    )
                metadata = {0: metadata}
        if metadata is None:
            metadata = {worker_id: {}}

    return {"data": data, AutoRestartBatchKeys.PL_RESTART_META: metadata}


# TODO: Merge this code within stateful DataLoaderIter.
def _next_data_wrapper(
    fn: Callable,
    it: Iterator,
    dl: DataLoader,
    num_batches_fetched: int,
    data_fetcher: "pl.utilities.fetching.AbstractDataFetcher",
) -> Callable:
    @wraps(fn)
    def wrapper() -> Any:
        nonlocal num_batches_fetched

        dataset = dl.dataset
        combined_batch = fn()

        batch, state = combined_batch["data"], combined_batch[AutoRestartBatchKeys.PL_RESTART_META]
        num_batches_fetched += 1

        if isinstance(dataset, CaptureIterableDataset):
            state = [
                IteratorState(
                    num_workers=dl.num_workers,
                    sampler_state=iterator_state,
                    num_batches_fetched=num_batches_fetched,
                    worker_id=list(iterator_state.keys())[0],
                    name=sampler_iter_name,
                )
                for sampler_iter_name, iterator_state in state.items()
            ]
        elif isinstance(dataset, CaptureMapDataset):
            ff_sampler = _find_fast_forward_samplers(dl)
            assert ff_sampler is not None
            state = [
                IteratorState(
                    num_workers=dl.num_workers,
                    sampler_state=ff_sampler.state_dict(num_batches_fetched),
                    dataset_state=state,
                    worker_id=list(state.keys())[0],
                    num_batches_fetched=num_batches_fetched,
                )
            ]
        data_fetcher._store_dataloader_iter_state(it, state)
        return batch

    return wrapper


def patch_dataloader_iterator(
    dataloader: DataLoader,
    iterator: Iterator,
    data_fetcher: "pl.utilities.fetching.AbstractDataFetcher",
    num_batches_fetched: int = 0,
) -> None:
    """Patches the iterator of a PyTorch dataloader by injecting logic for fault-tolerant training when it is
    necessary to remove the sampler state dict from provided data batch.

    The custom data has this format:
    .. code-block:: python
        {
            "batch": ...,  # data returned by DataLoader
            "__pl_restart_meta": {
                "sampler0": {
                    0: {"current_iteration": ...},
                    1: {"current_iteration": ...},
                },
                "sampler1": ...,
            },
        }
    Each sampler in the worker process tracks the current iteration. We return all of them to the main process
    as part of the sample and then a special collate function :func:`_capture_metadata_collate`
    will extract the current iteration as part of the metadata returned by a custom batch.
    """

    if not _FaultTolerantMode.detect_current_mode().is_automatic:
        return

    assert isinstance(dataloader.dataset, (CaptureMapDataset, CaptureIterableDataset))
    iterator._next_data = _next_data_wrapper(
        iterator._next_data, iterator, dataloader, num_batches_fetched, data_fetcher
    )


def _add_capture_metadata_collate(dataloader: DataLoader) -> None:
    """Wrap default collate function to retrieve captured dataset state dict when fault tolerant is enabled."""
    fault_tolerant_mode = _FaultTolerantMode.detect_current_mode()
    collate_fn = dataloader.collate_fn
    if not fault_tolerant_mode.is_enabled or (
        isinstance(collate_fn, partial) and collate_fn.func is _capture_metadata_collate
    ):
        return
    dataloader.collate_fn = partial(
        _capture_metadata_collate,
        dataset=dataloader.dataset,
        collate_fn=collate_fn,
        fault_tolerant_mode=fault_tolerant_mode,
    )


def _reload_dataloader_state_dict_automatic_map_dataset(dataloader: DataLoader, state_dict: Dict[str, Any]) -> None:
    iterator_state = state_dict["state"][0]

    if not isinstance(iterator_state, IteratorState):
        iterator_state = IteratorState.from_state_dict(iterator_state)

    # reload sampler state
    ff_sampler = _find_fast_forward_samplers(dataloader)
    assert ff_sampler is not None
    ff_sampler.load_state_dict(iterator_state.sampler_state)

    # reload dataset state
    dataloader.dataset.load_state_dict(
        iterator_state.dataset_state,
        latest_worker_id=state_dict["latest_worker_id"],
        num_workers=iterator_state.num_workers,
    )


def _reload_dataloader_state_dict_automatic_iterable_dataset(
    dataset: CaptureIterableDataset, state_dict: Dict[str, Any]
) -> None:
    dataset.load_state_dict(
        {sampler_name: state[0]["sampler_state"] for sampler_name, state in state_dict["state"].items()}
    )


def _reload_dataloader_state_dict_automatic(dataloader: DataLoader, state_dict: Dict[str, Any]) -> None:
    dataset = dataloader.dataset
    if isinstance(dataset, CaptureMapDataset):
        _reload_dataloader_state_dict_automatic_map_dataset(dataloader, state_dict)

    elif isinstance(dataset, CaptureIterableDataset):
        _reload_dataloader_state_dict_automatic_iterable_dataset(dataset, state_dict)

    else:
        raise MisconfigurationException("This shouldn't be happening. Please, open an issue.")


def _reload_dataloader_state_dict_manual(dataloader: DataLoader, state_dict: Dict[str, Any]) -> None:
    # In manual mode, we don't wrap the user objects with `CaptureMapDataset` or `CaptureIterableDataset`
    # therefore, we need to reload the states manually.
    latest_worker_id = state_dict["latest_worker_id"]
    num_workers = state_dict["state"][latest_worker_id]["num_workers"]
    sampler_state = state_dict["state"][latest_worker_id].get("sampler_state", None)
    if sampler_state:
        # `sampler_state` keys contain all the DataLoader attribute names
        # which matched `_Stateful` API interface while collecting the `state_dict`.
        for dataloader_attr_name in sampler_state:
            obj = getattr(dataloader, dataloader_attr_name)
            if not isinstance(obj, _Stateful):
                raise MisconfigurationException(
                    f"The DataLoader attribute {dataloader_attr_name}:{obj} should have a `load_state_dict` method."
                )

            obj.load_state_dict(sampler_state[dataloader_attr_name])

    if not isinstance(dataloader.dataset, _Stateful):
        return

    dataset_state = {
        worker_id: state_dict["state"][worker_id]["dataset_state"][worker_id]
        for worker_id in state_dict["state"].keys()
    }

    dataloader.dataset.load_state_dict(_rotate_worker_indices(dataset_state, latest_worker_id, num_workers))


def _reload_dataloader_state_dict(dataloader: DataLoader, state_dict: Dict[str, Any]) -> None:
    """Utility to reload state_dict within dataloader for fault tolerance."""

    fault_tolerant_mode = _FaultTolerantMode.detect_current_mode()

    if not fault_tolerant_mode.is_enabled:
        return

    if fault_tolerant_mode.is_automatic:
        _reload_dataloader_state_dict_automatic(dataloader, state_dict)

    elif fault_tolerant_mode.is_manual:
        _reload_dataloader_state_dict_manual(dataloader, state_dict)

    else:
        raise MisconfigurationException("This shouldn't be happening. Please, open an issue.")


def _rotate_worker_indices(state: Dict[int, Any], latest_worker_id: int, num_workers: int) -> Dict[int, Any]:
    """This function is used to rotate the worker indices based on the `latest_worker_id` the training failed
    on."""
    if num_workers == 0:
        return state
    if latest_worker_id > num_workers - 1:
        raise MisconfigurationException("The `latest_worker_id` should be within [0, num_workers - 1].")
    if len(state) != num_workers:
        raise MisconfigurationException("The `state` should contain `num_workers - 1` values.")
    next_worker_id = latest_worker_id + 1
    old_to_new_worker_id_map = [((next_worker_id + i) % num_workers, i) for i in range(num_workers)]
    return {new_id: state[old_id] for old_id, new_id in old_to_new_worker_id_map if old_id in state}


class _StatefulDataLoaderIter(_BaseDataLoaderIter):
    """This mixin is used to make PyTorch DataLoaderIter stateful."""

    def __accumulate_state(self, sampler_state: Dict[int, Any]) -> None:
        # store sampler state within a queue alongside its idx.
        self._sampler_state_idx: int = getattr(self, "_sampler_state_idx", 0) + 1
        self._sampler_state.append((sampler_state, self._sampler_state_idx))

    def _store_sampler_state(self) -> None:
        """This function is used to extract the sampler states if any."""
        sampler_state: Dict[int, Any] = {
            k: v.state_dict()  # type: ignore[misc]
            for k, v in self._loader.__dict__.items()
            if isinstance(v, _Stateful) and k != "dataset"
        }
        self.__accumulate_state(sampler_state)

    def _next_index(self) -> Any:
        indexes = super()._next_index()
        self._store_sampler_state()
        return indexes

    def _prepare_loader(self, loader: DataLoader) -> None:
        _add_capture_metadata_collate(loader)
        self._loader = loader
        self._data_fetcher: "pl.utilities.fetching.AbstractDataFetcher" = loader._lightning_fetcher
        self.num_batches_fetched = 0
        self._sampler_state: List[Tuple[Dict[int, Any], int]] = []
        self._sampler_state_idx = 0

    def __del__(self) -> None:
        if isinstance(self._loader.collate_fn, partial):
            self._loader.collate_fn = self._loader.collate_fn.keywords["collate_fn"]

    def _next_data(self) -> Any:
        combined_batch = super()._next_data()

        batch, state = combined_batch["data"], combined_batch[AutoRestartBatchKeys.PL_RESTART_META]

        self.num_batches_fetched += 1

        sampler_state, sampler_state_idx = self._sampler_state.pop(0)
        # there is no workers within the samplers
        worker_id = list(state.keys())[0]

        state = [
            IteratorState(
                num_workers=self._loader.num_workers,
                sampler_state=sampler_state,
                dataset_state=state,
                worker_id=worker_id,
                num_batches_fetched=self.num_batches_fetched,
            )
        ]
        # ensures there is an alignment between the sampler state and currently fetched batch
        assert sampler_state_idx == self.num_batches_fetched
        self._data_fetcher._store_dataloader_iter_state(self, state)
        return batch


class _SingleProcessDataLoaderIterStateful(_StatefulDataLoaderIter, _SingleProcessDataLoaderIter):
    def __init__(self, loader: DataLoader):
        self._prepare_loader(loader)
        super().__init__(loader)


class _MultiProcessingDataLoaderIterStateful(_StatefulDataLoaderIter, _MultiProcessingDataLoaderIter):
    def __init__(self, loader: DataLoader):
        self._prepare_loader(loader)
        super().__init__(loader)


def _get_iterator(self: DataLoader) -> "_BaseDataLoaderIter":
    if not hasattr(self, "_lightning_fetcher"):
        raise MisconfigurationException(
            "A stateful iterator should be used only when a DataFetcher has been attached to the DataLoader."
        )
    if self.num_workers == 0:
        return _SingleProcessDataLoaderIterStateful(self)
    else:
        if hasattr(self, "check_worker_number_rationality"):
            self.check_worker_number_rationality()
        return _MultiProcessingDataLoaderIterStateful(self)


def _patch_dataloader_get_iterators() -> None:
    """This function is used to replace the DataLoader iterator by their stateful version."""
    if not _FaultTolerantMode.detect_current_mode().is_manual:
        return
    if not hasattr(DataLoader, "_ori_get_iterator"):
        DataLoader._ori_get_iterator = DataLoader._get_iterator
    DataLoader._get_iterator = _get_iterator  # type: ignore[assignment]


def _teardown_dataloader_get_iterators() -> None:
    """This function is used to restore the DataLoader `get_iterator` with its original one."""
    # cleanup the get_iterator replacement in case of Fault-tolerance.
    get_iterator = getattr(DataLoader, "_ori_get_iterator", None)
    if get_iterator:
        DataLoader._get_iterator = get_iterator  # type: ignore[assignment]
        del DataLoader._ori_get_iterator


def _validate_iterable_dataset(dataloader: DataLoader) -> None:
    SUPPORTED_SAMPLERS = (RandomSampler, SequentialSampler, DistributedSampler)

    dataset = dataloader.dataset

    if getattr(dataset, "__next__", None) is None:
        raise AttributeError(
            "Fault-tolerance doesn't support an `IterableDataset` without `__next__` "
            "method implemented. Hint: We recommend you to move your logic from `__iter__`"
            " inside and rely on a sampler to perform the sample sampling."
        )

    samplers = {k: v for k, v in dataset.__dict__.items() if isinstance(v, Sampler)}

    if not samplers:
        raise TypeError("Fault-tolerance doesn't support an IterableDataset without a sampler as attribute.")

    sampler = [v for v in samplers.values() if type(v) in SUPPORTED_SAMPLERS]

    if not sampler:
        raise TypeError(f"Fault-tolerance supports only {SUPPORTED_SAMPLERS}.")

    if len(sampler) > 1:
        raise ValueError(f"A single sampler is supported within an Iterable Dataset. Found {sampler}.")

    if type(sampler[0]) is DistributedSampler and sampler.shuffle:
        raise TypeError("A `DistributedSampler` sampler shuffle attribute is set to True.")
    elif type(sampler[0]) is not SequentialSampler:
        raise TypeError("Only `SequentialSampler` is supported.")


def _validate_map_dataset(dataloader: DataLoader) -> None:
    SUPPORTED_SAMPLERS = (RandomSampler, SequentialSampler, DistributedSampler)

    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and type(sampler) not in SUPPORTED_SAMPLERS:
        raise TypeError(f"Fault-tolerance supports only {SUPPORTED_SAMPLERS}.")

    if type(sampler) is DistributedSampler and sampler.shuffle:
        raise TypeError("A `DistributedSampler` sampler shuffle attribute is set to True.")
    elif type(sampler) is RandomSampler:
        raise TypeError("Only `SequentialSampler` is supported.")


def _validate_fault_tolerant_automatic(dataloader: Iterable, stage: "pl.trainer.states.RunningStage") -> None:
    """This function is used to validate that Fault-tolerance is possible with the user data."""
    if not _FaultTolerantMode.detect_current_mode().is_automatic:
        return

    from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator

    if isinstance(dataloader, CombinedLoader):
        dataloaders = dataloader.loaders
    else:
        dataloaders = dataloader

    dl_loaders = []

    def flatten_dataloader(dataloader: Union[DataLoader, CycleIterator, Iterable]) -> None:
        nonlocal dl_loaders
        if isinstance(dataloader, CycleIterator):
            dataloader = dataloader.loader
        dl_loaders.append(dataloader)

    apply_to_collection(dataloaders, (DataLoader, CycleIterator), flatten_dataloader)

    if len(dl_loaders) > 1 and stage == pl.trainer.states.RunningStage.TRAINING:
        raise ValueError("Fault-tolerance supports only a single dataloader.")

    for dataloader in dl_loaders:
        assert isinstance(dataloader, DataLoader)
        validator_fn = (
            _validate_iterable_dataset if isinstance(dataloader.dataset, IterableDataset) else _validate_map_dataset
        )
        validator_fn(dataloader)


def _collect_states_on_rank_zero_over_collection(state_dict: Dict, key: str = "state") -> Dict:
    """This utility collects the state across processes for a collection of state."""

    def fn(state: Dict) -> Dict:
        if key in state:
            return _collect_states_on_rank_zero(state)
        return {k: apply_to_collection(v, Dict, fn) for k, v in state.items()}

    return apply_to_collection(state_dict, Dict, fn)
