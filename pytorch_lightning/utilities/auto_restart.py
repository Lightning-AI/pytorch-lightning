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

from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial, wraps
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info, Sampler
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.enums import AutoRestartBatchKeys
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training


class FastForwardSampler(Sampler):
    """This FastForwardSampler wraps a :class:`torch.utils.data.Sampler` and records the number of iterations
    performed during an epoch.

    It maintains a state, saved with :meth:`state_dict`, that can be reloaded with
    :meth:`load_state_dict`. If the sampler is used in a multiprocessing context, the ``FastForwardSampler`` will record
    the state of the current worker.
    When reloading, the ``FastForwardSampler`` will "fast-forward" the wrapped sampler by iterating through all the
    samples seen in the last iterations (for the current worker).
    """

    def __init__(self, sampler: Union[Sampler, Generator], attr_name: Optional[str] = None) -> None:
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

    def __next__(self):
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
    def from_state_dict(cls, state_dict) -> "IteratorState":
        return cls(**state_dict)


@dataclass
class MergedIteratorState:
    """This class is used to hold the current iterator state and lives on the iterator.

    It holds the current merged states from all worker processes. Once an iterator advances, it can store updates of the
    worker states in this merged iterator state.
    """

    state: Union[Dict[Union[int, str], Union[Dict[str, IteratorState], IteratorState]]] = field(default_factory=dict)
    latest_worker_id: int = 0
    represent_map_dataset: Optional[bool] = None

    def update(self, generator_name: Optional[str], new_state: IteratorState) -> None:
        # a map based dataset doesn't own a generator and therefore `generator_name` should be None.
        self.represent_map_dataset = generator_name is None
        if self.represent_map_dataset:
            state = self.state
        else:
            if generator_name not in self.state:
                self.state[generator_name] = {}
            state = self.state[generator_name]

        latest_worker_id = new_state.worker_id
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
    def from_state_dict(cls, state_dict) -> "MergedIteratorState":
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
        self.dataset = dataset
        self._cached_state_dict = None

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    def __getitem__(self, item) -> Tuple[Any, Dict[int, Dict]]:
        if self._cached_state_dict is not None:
            if self.worker_id in self._cached_state_dict:
                set_rng_states(self._cached_state_dict[self.worker_id]["rng_states"])
            self._cached_state_dict = None

        return self.dataset[item]

    def __len__(self) -> int:
        return len(self.dataset)

    def load_state_dict(self, state_dict: Dict[int, Any], latest_worker_id: int, num_workers: int) -> None:
        # as workers aren't available, the ``state_dict``` is cached until workers are made available.
        state_dict = deepcopy(state_dict)

        if num_workers > 0:
            # remap states to worker ids starting at 0
            next_worker_id = latest_worker_id + 1
            old_to_new_worker_id_map = [((next_worker_id + i) % num_workers, i) for i in range(num_workers)]
            state_dict = {
                new_id: state_dict[old_id] for old_id, new_id in old_to_new_worker_id_map if old_id in state_dict
            }
        self._cached_state_dict = state_dict

    def state_dict(self) -> Dict[int, Dict[str, Any]]:
        return {self.worker_id: {"rng_states": collect_rng_states()}}


def collect_rng_states() -> Dict[str, Any]:
    """Collect the global random state of :mod:`torch`, :mod:`numpy` and Python."""
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": python_get_rng_state()}


def set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    """Set the global random state of :mod:`torch`, :mod:`numpy` and Python in the current process."""
    torch.set_rng_state(rng_state_dict.get("torch"))
    np.random.set_state(rng_state_dict.get("numpy"))
    version, state, gauss = rng_state_dict.get("python")
    python_set_rng_state((version, tuple(state), gauss))


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
        self._state_dict: Optional[Dict[int, Any]] = None
        self._has_wrapped: bool = False

    @property
    def sampler(self) -> Sampler:
        return self.dataset.sampler

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.state_dict() for k, v in self.samplers.items()}

    def load_state_dict(self, state_dict: Dict[int, Any]) -> None:
        self._state_dict = deepcopy(state_dict)

    def _wrap_generator_samplers(self) -> None:
        self.samplers = {}

        # access wrapped dataset attributes
        dataset_dict = self.dataset.__dict__

        # create a tuple of sampler names
        samplers_names = tuple(v.__class__.__name__ for k, v in dataset_dict.items() if isinstance(v, Sampler))

        # create a dictionary of generator present within the dataset attributes
        dataset_sampler_generators = {k: v for k, v in dataset_dict.items() if isinstance(v, (Generator, Iterator))}

        # iterate over the generator. If a generator was created from a `Sampler`,
        # it will be wrapped into a `FastForwardSampler`.
        for (generator_attr_name, generator) in dataset_sampler_generators.items():

            if isinstance(generator, Sampler):
                continue

            # used to handle a weird behaviour from PyTorch 1.6
            # where the sampler is converted to a list_iterator
            is_legacy = False

            if isinstance(generator, Generator):
                # Generator name have the  the form `SamplerName.__iter__`
                generator_name = generator.__qualname__.split(".")[0]
            else:
                # assume the retrieved iterator is coming from sampler.
                is_legacy = True

            # validate the base generator name matches a sampler name.
            if is_legacy or any(sampler_name == generator_name for sampler_name in samplers_names):

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

    def reset_on_epoch(self):
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
    # the `num_workers` for fault tolerant training
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

        # we can finally call reset and apply prefecthing.
        iter_dataloader._reset = iter_dataloader._original_reset
        iter_dataloader._reset(dataloader, first_iter=True)
    # return the iterator
    return iter_dataloader


def _dataloader_to_state_dict(
    dataloader: DataLoader, iterator: Iterator, num_batches_processed: int = None
) -> List[Dict[str, Any]]:
    """Convert a dataloader to its associated state dict."""
    out = {}
    if iterator is not None:
        out.update(_find_current_worker(iterator))

    if not isinstance(dataloader.dataset, CaptureIterableDataset):
        fast_forward_sampler = _find_fast_forward_samplers(dataloader)
        if fast_forward_sampler is not None:
            out.update(fast_forward_sampler.state_dict(num_batches_processed=num_batches_processed))
    return out


def _dataloader_load_state_dict(dataloader: DataLoader, state_dict: List[Dict[str, Any]]) -> DataLoader:
    """Reload ``DataLoader`` fast-forward sampler state dict."""
    fast_forward_sampler = _find_fast_forward_samplers(dataloader)

    if isinstance(fast_forward_sampler, Sampler):
        state_dict = {k: v for k, v in state_dict.items() if k not in ("num_workers", "previous_worker")}
        fast_forward_sampler.load_state_dict(state_dict)

    return dataloader


def _find_current_worker(iterator: Iterator) -> Dict[str, Optional[int]]:
    """Find the current DataLoader Iterator worker if multiple workers were used."""
    # get the current number of workers
    num_workers = getattr(iterator, "_num_workers", 0)
    if isinstance(iterator, _MultiProcessingDataLoaderIter):
        # fetch next worker
        next_worker = (next(iterator._worker_queue_idx_cycle)) % num_workers
        # get the current worker from next one
        previous_worker = (next_worker - 1) % num_workers
        # reset back the `worker_queue_idx` to current one, so we can keep
        # going without perturbation.
        while next(iterator._worker_queue_idx_cycle) != previous_worker:
            pass
    else:
        previous_worker = None

    # return the captured metadata.
    return {"num_workers": num_workers, "previous_worker": previous_worker}


def _capture_metadata_collate(samples: List, dataset: Dataset, default_collate: Callable) -> Dict:
    """A collate function that adds the state dict of a :class:`CaptureIterableDataset` or
    :class:`CaptureMapDataset` used in the worker processes. This function gets executed within the worker
    processes. The structure will be:

    .. code-block:: python

        {
            "data": ...,  # data returned by Dataset
            "__pl_restart_meta": {"sampler_name0": state_dict0, "sampler_name1": state_dict1},
        }
    """
    data = default_collate(samples)
    if not isinstance(dataset, (CaptureIterableDataset, CaptureMapDataset)):
        return data
    metadata = dataset.state_dict()
    return {"data": data, AutoRestartBatchKeys.PL_RESTART_META: metadata}


def patch_dataloader_iterator(
    dataloader: DataLoader,
    iterator: Iterator,
    data_fetcher: "pl.utilities.fetching.DataFetcher",
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

    assert isinstance(dataloader.dataset, (CaptureMapDataset, CaptureIterableDataset))

    def _next_data_wrapper(fn, it, dl, num_batches_fetched) -> Callable:
        @wraps(fn)
        def wrapper():
            nonlocal num_batches_fetched
            nonlocal it
            nonlocal dl

            dataset = dl.dataset
            combined_batch = fn()

            batch, state = combined_batch["data"], combined_batch[AutoRestartBatchKeys.PL_RESTART_META]
            num_batches_fetched += 1

            if isinstance(dataset, CaptureIterableDataset):
                state = [
                    IteratorState(
                        num_workers=dataloader.num_workers,
                        sampler_state=iterator_state,
                        num_batches_fetched=num_batches_fetched,
                        worker_id=list(iterator_state.keys())[0],
                        name=sampler_iter_name,
                    )
                    for sampler_iter_name, iterator_state in state.items()
                ]
            elif isinstance(dataset, CaptureMapDataset):
                ff_sampler = _find_fast_forward_samplers(dl)
                state = [
                    IteratorState(
                        num_workers=dataloader.num_workers,
                        sampler_state=ff_sampler.state_dict(num_batches_fetched),
                        dataset_state=state,
                        worker_id=list(state.keys())[0],
                        num_batches_fetched=num_batches_fetched,
                    )
                ]
            data_fetcher._store_dataloader_iter_state(it, state)
            return batch

        return wrapper

    iterator._next_data = _next_data_wrapper(iterator._next_data, iterator, dataloader, num_batches_fetched)


def _add_capture_metadata_collate(dataloader: DataLoader) -> None:
    """Wrap default collate function to retrive captured dataset state dict when fault tolerant is enabled."""
    dataloader.collate_fn = partial(
        _capture_metadata_collate, dataset=dataloader.dataset, default_collate=dataloader.collate_fn
    )


def reload_dataloader_state_dict(dataloader: DataLoader, state_dict: Dict[str, Any]) -> None:
    """Utility to reload state_dict within dataloader for fault tolerance."""

    if not _fault_tolerant_training():
        return

    dataset = dataloader.dataset

    if isinstance(dataset, CaptureMapDataset):
        iterator_state = state_dict["state"][0]

        if not isinstance(iterator_state, IteratorState):
            iterator_state = IteratorState.from_state_dict(iterator_state)

        # reload sampler state
        ff_sampler = _find_fast_forward_samplers(dataloader)
        ff_sampler.load_state_dict(iterator_state.sampler_state)

        # reload dataset state
        dataset.load_state_dict(
            iterator_state.dataset_state,
            latest_worker_id=state_dict["latest_worker_id"],
            num_workers=iterator_state.num_workers,
        )

    elif isinstance(dataset, CaptureIterableDataset):
        dataset.load_state_dict(
            {sampler_name: state[0]["sampler_state"] for sampler_name, state in state_dict["state"].items()}
        )

    else:
        raise MisconfigurationException("This shouldn't happen. Please, open an issue on PyTorch Lightning Github.")
