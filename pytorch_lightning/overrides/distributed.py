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
import itertools
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DistributedSampler, get_worker_info, Sampler
from torch.utils.data.dataloader import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.enums import BatchKeys


class LightningDistributedModule(_LightningModuleWrapperBase):

    def __init__(self, pl_module: 'pl.LightningModule') -> None:
        """
        Wraps the user's LightningModule and redirects the forward call to the appropriate
        method, either ``training_step``, ``validation_step``, ``test_step`` or ``predict``.
        This class is used in combination with :class:`~torch.nn.parallel.DistributedDataParallel` as
        shown in the example.

        Example:

            ddp_model = torch.nn.parallel.DistributedDataParallel(
                module=LightningDistributedModule(lightning_module),
                device_ids=[local_rank],
                ...
            )

        Args:
            pl_module: the model to wrap

        """
        super().__init__(pl_module)


def _find_tensors(obj):  # pragma: no-cover
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


# In manual_optimization, we need to call reducer prepare_for_backward.
# Note: Keep track of Pytorch DDP and update if there is a change
# https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/parallel/distributed.py#L626-L638
def prepare_for_backward(model: DistributedDataParallel, output: Any):
    # `prepare_for_backward` is `DistributedDataParallel` specific.
    if not isinstance(model, DistributedDataParallel):
        return
    if torch.is_grad_enabled() and model.require_backward_grad_sync:
        model.require_forward_param_sync = True
        # We'll return the output object verbatim since it is a freeform
        # object. We need to find any tensors in this object, though,
        # because we need to figure out which parameters were used during
        # this forward pass, to ensure we short circuit reduction for any
        # unused parameters. Only if `find_unused_parameters` is set.
        if model.find_unused_parameters:
            model.reducer.prepare_for_backward(list(_find_tensors(output)))
        else:
            model.reducer.prepare_for_backward([])
    else:
        model.require_forward_param_sync = False


class UnrepeatedDistributedSampler(DistributedSampler):
    """
    A fork of the pytorch DistributedSampler that doesn't repeat data, instead
    allowing the number of batches per process to be off-by-one from each other.
    This makes this sampler usable for predictions (it's deterministic and
    doesn't require shuffling). It is potentially unsafe to use this sampler for
    training, because during training the DistributedDataParallel syncs buffers
    on each forward pass, so it could freeze if one of the processes runs one
    fewer batch. During prediction, buffers are only synced on the first batch,
    so this is safe to use as long as each process runs at least one batch. We
    verify this in an assert.

    Taken from https://github.com/jpuigcerver/PyLaia/blob/v1.0.0/laia/data/unpadded_distributed_sampler.py
    and https://github.com/pytorch/pytorch/issues/25162#issuecomment-634146002
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_samples = len(range(self.rank, len(self.dataset), self.num_replicas))
        self.total_size = len(self.dataset)
        # If any process has at least one batch, every other process needs to
        # have at least one batch, or the DistributedDataParallel could lock up.
        assert self.num_samples >= 1 or self.total_size == 0

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class IndexBatchSamplerWrapper:
    """This class is used to wrap a :class:`torch.utils.data.BatchSampler` and capture its indices."""

    def __init__(self, sampler: BatchSampler) -> None:
        self._sampler = sampler
        self.batch_indices: Optional[List[int]] = None

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._sampler:
            self.batch_indices = batch
            yield batch

    def __len__(self) -> int:
        return len(self._sampler)

    @property
    def drop_last(self) -> bool:
        return self._sampler.drop_last

    @property
    def batch_size(self) -> int:
        return self._sampler.batch_size

    @property
    def sampler(self) -> Sampler:
        return self._sampler.sampler


class FastForwardSampler:
    """This class is used to wrap a :class:`torch.utils.data.Sampler` and fast forward it"""

    def __init__(self, sampler: Union[Sampler, BatchSampler, Generator]) -> None:
        self._sampler = sampler
        self._current_iteration = 0
        self.rng_state: Optional[torch.Tensor] = None
        self._num_workers: Optional[int] = None
        self.restarting: bool = False
        self.inside_workers: Optional[bool] = None
        self._state_dict: Optional[Dict[str, Any]] = None

    def setup(self, num_workers: int, batch_size: int, inside_workers: bool) -> None:
        self._num_workers = num_workers
        self.current_batch_size = batch_size
        self.inside_workers = inside_workers

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    def __iter__(self) -> Iterator[List[int]]:
        if self._num_workers is None:
            raise Exception(
                f"The {self.__class__.__name__} hasn't been properly setup. Please, open an issue on PyTorch Lightning."
            )
        if self.rng_state is None:
            self.rng_state = torch.random.get_rng_state()
        local_counter = 0
        for batch in self._sampler:
            if self.restarting:
                if self._current_iteration == 0:
                    self.restarting = False
                if self._state_dict is not None and self.worker_id in self._state_dict:
                    worker_state_dict = self._state_dict[self.worker_id]
                    self.load_state_dict(worker_state_dict)
                    self._state_dict = None
                local_counter += 1
                if local_counter == self._current_iteration:
                    self.restarting = False
            else:
                if self.no_worker or self.inside_workers:
                    self._current_iteration += 1
                yield batch
        self.rng_state = None
        self._current_iteration = 0

    @property
    def no_worker(self):
        return self.num_workers == 0

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @num_workers.setter
    def num_workers(self, num_workers: int) -> None:
        self._num_workers = num_workers

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @current_iteration.setter
    def current_iteration(self, current_iteration: int) -> None:
        self._current_iteration = current_iteration

    def __len__(self) -> int:
        return len(self._sampler)

    @property
    def drop_last(self) -> bool:
        return self._sampler.drop_last

    @property
    def batch_size(self) -> int:
        return self._sampler.batch_size

    @property
    def sampler(self) -> Sampler:
        return self._sampler

    @property
    def batch_indices(self) -> Optional[List[int]]:
        return self._sampler.batch_indices

    def _compute_current_iteration(self, number_batch_processed: Optional[int] = None) -> int:
        if not (self.no_worker or self.inside_workers) and number_batch_processed is None:
            raise Exception("``number_batch_processed`` should be provided")

        if self.no_worker or self.inside_workers:
            current_iteration = self.current_iteration
        else:
            current_iteration = number_batch_processed

        if not self.inside_workers:
            current_iteration *= self.current_batch_size

        return current_iteration

    def state_dict(self, number_batch_processed: Optional[int] = None):
        rng_stage = self.rng_state
        if self.inside_workers and self.rng_state is None:
            rng_stage = torch.random.get_rng_state()
        return {"rng_state": rng_stage, "current_iteration": self._compute_current_iteration(number_batch_processed)}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if all(isinstance(k, int) for k in state_dict.keys()):
            self._state_dict = state_dict
            self.restarting = True
            return
        self.current_iteration = state_dict["current_iteration"]
        if state_dict["rng_state"] is not None:
            torch.random.set_rng_state(state_dict["rng_state"])
            self.rng_state = state_dict["rng_state"]
        self.restarting = True


class CaptureIterativeDataset(IterableDataset):

    def __init__(
        self,
        dataset: IterableDataset,
        num_workers: int,
        batch_size: int,
        is_inside_workers: bool,
        initial_seed: Optional[torch.Generator] = None
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.is_inside_workers = is_inside_workers
        self.samplers_state_dict: Optional[Dict[int, Any]] = None
        self.initial_seed = initial_seed
        self.samplers = None

    def setup(self, samplers_state_dict: Dict[int, Any]):
        self.samplers_state_dict = samplers_state_dict

    def _wrap_samplers(self) -> None:
        if self.samplers is None:
            self.samplers = {}
            dataset_dict = self.dataset.__dict__
            dataset_sampler_generators = {
                k: v
                for k, v in dataset_dict.items() if isinstance(v, Generator) and "sampler" in v.__qualname__.lower()
            }
            for (attr_name, sampler) in dataset_sampler_generators.items():
                sampler = FastForwardSampler(sampler)
                sampler.setup(self.num_workers, self.batch_size, self.is_inside_workers)
                if self.samplers_state_dict is not None:
                    sampler.load_state_dict(self.samplers_state_dict[attr_name])
                self.samplers[attr_name] = sampler
                dataset_dict[attr_name] = iter(sampler)

    @property
    def sampler(self) -> Sampler:
        return self.dataset.sampler

    def __iter__(self) -> Iterator:
        self.iter_data = iter(self.dataset)
        self._wrap_samplers()
        return self

    def __next__(self):
        data = next(self.iter_data)
        worker_info = get_worker_info()
        state_dicts = {"id": worker_info.id if worker_info is not None else 0}
        for k, v in self.samplers.items():
            state_dicts.update({k: v.state_dict()})
        return {"data": data, BatchKeys.PL_SAMPLERS: state_dicts}


def _find_fast_forward_samplers(dataloader: Union[DataLoader, CombinedLoader], state_dict):
    if isinstance(dataloader, CombinedLoader):
        dataloader = dataloader.loaders
    dataset = dataloader.dataset
    if isinstance(dataset, CaptureIterativeDataset):
        return dataset.setup(state_dict)
    else:
        sampler = dataloader.sampler
        return sampler if isinstance(sampler, FastForwardSampler) else dataloader.batch_sampler
