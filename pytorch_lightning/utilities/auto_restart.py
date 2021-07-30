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
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Union

from torch.utils.data import Dataset, get_worker_info, Sampler
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, DataLoader, IterableDataset

from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.enums import AutoRestartBatchKeys
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class FastForwardSampler(Sampler):
    """
    This FastForwardSampler wraps a :class:`torch.utils.data.Sampler` and records the number of iterations
    performed during an epoch. It maintains a state, saved with :meth:`state_dict`, that can be reloaded with
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
        self._dataloader_batch_size: Optional[int] = None
        self._cached_state_dict: Optional[Dict[str, Any]] = None
        self._attr_name = attr_name

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        return getattr(self._sampler, key, None)

    def setup(self, dataloader_batch_size: Optional[int] = None) -> None:
        """
        Setup the ``FastForwardSampler``.
        This is required only when the provided dataset subclassed :class:`torch.utils.data.Dataset`.
        """
        self._dataloader_batch_size = dataloader_batch_size

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    def __iter__(self) -> Iterator[Any]:
        # split restart logic to avoid user with tempering with "fast-forwarding"

        if not self.restarting:
            for batch in self._sampler:
                self._current_iteration += 1
                yield batch

        else:
            for i, batch in enumerate(self._sampler):

                # the `state dict` was cached as workers were available before.
                if self._cached_state_dict is not None and self.worker_id in self._cached_state_dict:

                    # reload the current state dict
                    self.load_state_dict(self._cached_state_dict, workers_initialized=True)
                    self._cached_state_dict = None

                # when the current index matching the current_iteration, we have "fast forwarded" the sampler.
                if self._current_iteration <= i:
                    if self._current_iteration == i:
                        self.restarting = False
                    self._current_iteration += 1
                    yield batch

        self._current_iteration = 0
        self.restarting = False

    def __len__(self) -> int:
        return len(self.sampler)

    def _compute_current_iteration(self, num_batches_processed: Optional[int] = None) -> int:
        """
        This function is used to compute the effective iteration.
        As DataLoader can perform ``prefecthing`` or training can fail while processing a batch,
        the current iteration needs to be computed using the ``num_batches_processed`` processed information.
        """
        if num_batches_processed is not None:
            current_iteration = num_batches_processed
        else:
            current_iteration = self._current_iteration

        if self._dataloader_batch_size and num_batches_processed is not None:
            current_iteration *= self._dataloader_batch_size

        return current_iteration

    def state_dict(self, num_batches_processed: Optional[int] = None) -> Dict[int, Dict[str, int]]:
        """Returns the state of the sampler in the current worker. The worker id indexes the state dict."""
        return {self.worker_id: {"current_iteration": self._compute_current_iteration(num_batches_processed)}}

    def load_state_dict(self, state_dict: Dict[int, Any], workers_initialized: bool = False) -> None:
        """
        Loads the saved state for the wrapped sampler.
        If the ``state_dict`` contains multiple states, it means there were multiple workers.
        The state will be cached and fully reloaded (fast-forward) the first time :meth:`__iter__` is called.
        """
        # as workers aren't available, the `state_dict` is cached until workers are made available.
        if len(state_dict) > 1 and not workers_initialized:
            self._cached_state_dict = deepcopy(state_dict)
            self.restarting = True
            return
        self._current_iteration = state_dict[self.worker_id]["current_iteration"]
        self.restarting = True


class CaptureIterableDataset(IterableDataset):
    """
    The ``CaptureIterableDataset`` is used to wrap an :class:`torch.utils.data.IterableDataset`.
    On ``__iter__`` function call,   the ``CaptureIterableDataset`` will wrap the wrapped dataset
        generators into ``FastForwardSampler`` to keep track of progress.
    On ``__next__`` function call, the ``CaptureIterableDataset`` will return a dictionary containing
        user data and metadata containing the ``FastForwardSampler`` samplers state_dict.
    """

    def __init__(self, dataset: IterableDataset, initial_seed: Optional[int] = None) -> None:
        super().__init__()
        self.dataset = deepcopy(dataset)
        self._state_dict: Optional[Dict[int, Any]] = None
        self.initial_seed = initial_seed
        self.samplers: Optional[Dict[str, FastForwardSampler]] = None

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

    def reset_on_epoch(self) -> None:
        self._state_dict = None

    def __iter__(self) -> Iterator:
        # create a generator from the wrapped Iterative Dataset
        # if the dataset contained samplers, they will be transformers into generators
        self.iter_data = iter(self.dataset)

        # wrap any generator associated to a Sampler into a `FastForwardSampler`.
        self._wrap_generator_samplers()
        return self

    def __next__(self) -> Dict[str, Any]:
        return next(self.iter_data)

    @staticmethod
    def store_samplers_state_dict(iterator: Iterator, sampler_state_dict: List) -> None:
        """
        This function is used to store and update sampler state dict on its associated iterator.

        In Lightning, as the iterator is wrapped into a prefetching function,
        we needed to introduce a cache to delay updating the ``sampler_state_dict``.
        """
        iterator_state_dict = getattr(iterator, "_sampler_state_dict", None)
        iterator_state_dict_cache = getattr(iterator, "_sampler_state_dict_cache", None)
        # handle the logic this way due Trainer prefetching.
        if iterator_state_dict is None:
            iterator._sampler_state_dict = sampler_state_dict
        elif iterator_state_dict_cache is None:
            iterator._sampler_state_dict_cache = sampler_state_dict
        else:
            for attr_cache, state_dict in zip(iterator_state_dict, iterator._sampler_state_dict_cache):
                for k, v in state_dict.items():
                    attr_cache[k].update(v)
            iterator._sampler_state_dict_cache = sampler_state_dict

    @staticmethod
    def _sanitize_batch_from_sampler_state(data: Any, state_dicts: List):
        """
        This function is used to remove the sampler state dict from provided data batch.
        The custom data has this format:

        .. code-block:: python

            {
                "batch": ...,  # data returned by DataLoader
                "__pl_samplers": {
                    "sampler0": {
                        0: {"current_iteration": ...},
                        1: {"current_iteration": ...},
                    },
                    "sampler1": ...,
                },
            }

        Each sampler in the worker process tracks the current iteration. We return all of them to the main process
        as part of the sample and then a special collate function :func:`_sampler_metadata_collate`
        will extract the current iteration as part of the metadata returned by a custom batch.
        """

        def _sanitize(data: Mapping):
            out = []
            for k, v in data.items():
                if k == AutoRestartBatchKeys.PL_SAMPLERS:
                    state_dicts.append(v)
                    return data["data"]
                out.append((k, CaptureIterableDataset._sanitize_batch_from_sampler_state(v, state_dicts)))
            return out

        return apply_to_collection(data, Mapping, _sanitize)

    @staticmethod
    def extract_samplers_state_dict_from_batch(batch) -> List[Dict[int, Any]]:
        """
        This function is used to convert a batch into a state_dict
        """
        samplers_state_dict = []

        batch = CaptureIterableDataset._sanitize_batch_from_sampler_state(batch, samplers_state_dict)

        return batch, samplers_state_dict


def _find_fast_forward_samplers(dataloader: DataLoader) -> Optional[FastForwardSampler]:
    """
    If the ``DataLoader`` is wrapping a mapping based Dataset, return the ``FastForwardSampler``.
    """
    if isinstance(dataloader.sampler, FastForwardSampler):
        return dataloader.sampler

    if isinstance(dataloader.batch_sampler, FastForwardSampler):
        return dataloader.batch_sampler


def _cycle_to_next_worker_and_reset(dataloader: DataLoader, state_dict: Dict[str, Any]) -> Iterator:
    """
    This function is used to cycle back the DataLoader ``_MultiProcessingDataLoaderIter``
    workers and call the reset function.

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
    """
    Convert a dataloader to its associated state dict
    """
    out = {}
    if iterator is not None:
        out.update(_find_current_worker(iterator))

    if not isinstance(dataloader.dataset, CaptureIterableDataset):
        fast_forward_sampler = _find_fast_forward_samplers(dataloader)
        if fast_forward_sampler is not None:
            out.update(fast_forward_sampler.state_dict(num_batches_processed=num_batches_processed))
    return out


def _dataloader_load_state_dict(dataloader: DataLoader, state_dict: List[Dict[str, Any]]) -> DataLoader:
    """
    Reload ``DataLoader`` fast-forward sampler state dict.
    """
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


def _sampler_metadata_collate(samples: List, dataset: Dataset, default_collate: Callable) -> Dict:
    """
    A collate function that adds the state dict of all samplers used in the worker processes.

    The structure will be:

    .. code-block:: python

        {
            "data": ...,  # data returned by Dataset
            "__pl_samplers": {"sampler_name0": state_dict0, "sampler_name1": state_dict1},
        }
    """
    batch = default_collate(samples)
    if not isinstance(dataset, CaptureIterableDataset):
        return batch
    return {"data": batch, AutoRestartBatchKeys.PL_SAMPLERS: dataset.state_dict()}
