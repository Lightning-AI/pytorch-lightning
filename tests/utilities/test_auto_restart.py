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
import math
import os
import random
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler, SequentialSampler
from torch.utils.data._utils.worker import _generate_state, get_worker_info
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.dataset import Dataset, IterableDataset

import tests.helpers.utils as tutils
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.auto_restart import CaptureIterativeDataset, FastForwardSampler
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.enums import AutoRestartBatchKeys
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.runif import RunIf


def parse_metadata(batch):
    return {
        k: {
            batch[AutoRestartBatchKeys.PL_SAMPLERS]["id"][-1].item(): {
                "current_iteration": v["current_iteration"][-1].item(),
                "rng_state": v["rng_state"][-1]
            }
        }
        for k, v in batch[AutoRestartBatchKeys.PL_SAMPLERS].items() if k != "id"
    }


def test_fast_forward_on_batch_sampler():
    """
    This test ensures ``FastForwardSampler`` applied to ``BatchSampler`` correctly retrived
    the right next batch on restart.
    """
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = FastForwardSampler(batch_sampler)
    index_batch_sampler.setup(1, 1, False)

    assert isinstance(index_batch_sampler, Iterable)
    assert has_len(index_batch_sampler)

    index_batch_sampler_iter = iter(index_batch_sampler)

    assert next(index_batch_sampler_iter) == [0, 1, 2]
    assert next(index_batch_sampler_iter) == [3, 4, 5]

    state_dict = index_batch_sampler.state_dict(2)

    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = FastForwardSampler(batch_sampler)
    index_batch_sampler.setup(1, 1, False)
    index_batch_sampler.load_state_dict(state_dict)

    index_batch_sampler_iter = iter(index_batch_sampler)
    assert next(index_batch_sampler_iter) == [6, 7, 8]


def test_fast_forward_on_sequential_sampler():
    """
    This test ensures ``FastForwardSampler`` applied to ``SequentialSampler`` correctly retrived
    the right next batch on restart.
    """
    dataset = range(15)
    sampler = FastForwardSampler(SequentialSampler(dataset))
    sampler.setup(1, 3, False)
    batch_sampler = BatchSampler(sampler, 3, False)

    batch_sampler_iter = iter(batch_sampler)

    assert next(batch_sampler_iter) == [0, 1, 2]
    assert next(batch_sampler_iter) == [3, 4, 5]

    state_dict = sampler.state_dict(2)
    assert state_dict["current_iteration"] == 6

    dataset = range(15)
    sampler = FastForwardSampler(SequentialSampler(dataset))
    sampler.setup(1, 3, False)
    batch_sampler = BatchSampler(sampler, 3, False)
    sampler.load_state_dict(state_dict)

    batch_sampler_iter = iter(batch_sampler)
    assert next(batch_sampler_iter) == [6, 7, 8]


def test_fast_forward_on_random_sampler():
    """
    This test ensures ``FastForwardSampler`` applied to ``RandomSampler`` correctly retrived
    the right next batch on restart.
    """
    seed_everything(42)

    dataset = range(15)
    sampler = FastForwardSampler(RandomSampler(dataset))
    sampler.setup(1, 3, False)
    batch_sampler = BatchSampler(sampler, 3, False)

    batch_sampler_iter = iter(batch_sampler)

    assert next(batch_sampler_iter) == [14, 9, 1]
    assert next(batch_sampler_iter) == [7, 11, 3]
    assert next(batch_sampler_iter) == [12, 8, 2]

    state_dict = sampler.state_dict(3)
    assert state_dict["current_iteration"] == 9
    state_dict["current_iteration"] = 6

    dataset = range(15)
    sampler = FastForwardSampler(RandomSampler(dataset))
    sampler.setup(1, 3, False)
    batch_sampler = BatchSampler(sampler, 3, False)
    sampler.load_state_dict(state_dict)

    batch_sampler_iter = iter(batch_sampler)
    assert next(batch_sampler_iter) == [12, 8, 2]
    has_raised = False
    try:
        for _ in range(5):
            next(batch_sampler_iter)
    except StopIteration:
        has_raised = True
        assert sampler.rng_state is None
        assert sampler.current_iteration == 0
        sampler.load_state_dict(sampler.state_dict(0))
    assert has_raised


class RangeIterativeDataset(IterableDataset):

    def __init__(self, data, num_workers: int, batch_size: int, is_in_workers: bool, state_dict=None):
        self.data = list(data)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_in_workers = is_in_workers
        self.state_dict = state_dict

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info and self.num_workers == 2:
            id = worker_info.id
            num_samples = len(self.data)
            if id == 0:
                self.data = list(self.data)[:num_samples // 2]
            else:
                self.data = list(self.data)[num_samples // 2:]
            self.user_sampler = RandomSampler(self.data)
        else:
            self.user_sampler = RandomSampler(self.data)

        sampler = FastForwardSampler(self.user_sampler)
        sampler.setup(self.batch_size, self.num_workers, self.is_in_workers)
        if self.state_dict is not None:
            sampler.load_state_dict(self.state_dict[0]["iter_sampler"])
            self.state_dict = None
        self.sampler = sampler
        self.iter_sampler = iter(self.sampler)
        return self

    def __next__(self):
        return self.data[next(self.iter_sampler)]


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_fast_forward_sampler_over_iterative_dataset(num_workers):
    """
    This test ensures ``FastForwardSampler`` and ``CaptureIterativeDataset`` are properly being
    used to capture workers states.
    """
    batch_size = 3
    initial_seed = seed_everything(42)
    generator = torch.Generator()
    generator.manual_seed(initial_seed)
    dataset = RangeIterativeDataset(range(20), num_workers, batch_size, True)
    dataset = CaptureIterativeDataset(dataset, num_workers, batch_size, True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=generator)
    iter_dataloader = iter(dataloader)
    batches = []
    for _ in range(5):
        batches.append(next(iter_dataloader))

    if num_workers == 0:
        batch_0_expected = torch.tensor([4, 3, 12])
        batch_1_expected = torch.tensor([18, 0, 7])
        batch_2_expected = torch.tensor([8, 14, 11])
        batch_3_expected = torch.tensor([5, 10, 13])
        batch_4_expected = torch.tensor([17, 19, 15])
    elif num_workers == 1:
        batch_0_expected = torch.tensor([3, 18, 17])
        batch_1_expected = torch.tensor([13, 2, 19])
        batch_2_expected = torch.tensor([6, 4, 7])
        batch_3_expected = torch.tensor([1, 14, 5])
        batch_4_expected = torch.tensor([12, 8, 16])
    else:
        batch_0_expected = torch.tensor([3, 4, 5])
        batch_1_expected = torch.tensor([10, 12, 14])
        batch_2_expected = torch.tensor([7, 0, 9])
        batch_3_expected = torch.tensor([16, 18, 17])
        batch_4_expected = torch.tensor([8, 1, 2])

    assert torch.equal(batches[0]["data"], batch_0_expected)
    assert torch.equal(batches[1]["data"], batch_1_expected)
    assert torch.equal(batches[2]["data"], batch_2_expected)
    assert torch.equal(batches[3]["data"], batch_3_expected)
    assert torch.equal(batches[4]["data"], batch_4_expected)

    # restarting on batch_1 and getting 3 extra batches

    state_dict = {0: {'iter_sampler': {}}}
    for batch in batches[:2]:
        metadata = parse_metadata(batch)
        for k, v in metadata.items():
            state_dict[0][k].update(v)

    if num_workers == 2:
        assert len(state_dict[0]["iter_sampler"]) == 2

    initial_seed = seed_everything(42)
    generator.manual_seed(initial_seed)
    dataset = RangeIterativeDataset(range(20), num_workers, batch_size, True, state_dict=state_dict)
    dataset = CaptureIterativeDataset(dataset, num_workers, batch_size, True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=generator)
    iter_dataloader = iter(dataloader)
    batches = []
    for _ in range(3):
        batches.append(next(iter_dataloader))

    assert torch.equal(batches[0]["data"], batch_2_expected)
    assert torch.equal(batches[1]["data"], batch_3_expected)
    assert torch.equal(batches[2]["data"], batch_4_expected)


def _setup_ddp(rank, worldsize):
    os.environ["MASTER_ADDR"] = "localhost"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def _test_fast_forward_sampler_with_distributed_sampler(rank, worldsize):
    _setup_ddp(rank, worldsize)

    initial_seed = seed_everything(42)

    generator = torch.Generator()
    generator.manual_seed(initial_seed)

    num_workers = 2
    batch_size = 4

    dataset = range(30)
    sampler = FastForwardSampler(
        DistributedSampler(dataset, num_replicas=worldsize, rank=rank, drop_last=True, seed=initial_seed)
    )
    sampler.setup(num_workers, batch_size, False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, generator=generator, sampler=sampler
    )

    iter_dataloader = iter(dataloader)

    num_yielded = 0
    batches = []
    while True:
        try:
            batches.append(next(iter_dataloader))
            num_yielded += 1
        except StopIteration:
            break

    expected = torch.tensor([17, 27, 24]) if rank == 0 else torch.tensor([19, 5, 3])
    assert torch.equal(batches[-1], expected)
    assert sampler.state_dict(num_yielded)["current_iteration"] == 16

    sampler = FastForwardSampler(
        DistributedSampler(dataset, num_replicas=worldsize, rank=rank, drop_last=True, seed=initial_seed)
    )
    sampler.setup(num_workers, batch_size, False)
    sampler.load_state_dict({'rng_state': None, 'current_iteration': 12})
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, generator=generator, sampler=sampler
    )

    iter_dataloader = iter(dataloader)

    batches = []
    while True:
        try:
            batches.append(next(iter_dataloader))
        except StopIteration:
            break

    assert torch.equal(batches[-1], expected)
    assert sampler.state_dict(num_yielded)["current_iteration"] == 16


@RunIf(skip_windows=True)
def test_fast_forward_sampler_with_distributed_sampler():
    """Make sure result logging works with DDP"""
    tutils.set_random_master_port()
    worldsize = 2
    mp.spawn(_test_fast_forward_sampler_with_distributed_sampler, args=(worldsize, ), nprocs=worldsize)


# Iterative Dataset


class MetaLearningDataset(IterableDataset):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        task_num_classes: int = 5,
        num_workers: Optional[int] = None,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        initial_seed: Optional[torch.Generator] = None,
        shuffle: bool = True,
        debugging: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_workers = num_workers or 1
        self.global_rank = global_rank
        self.world_size = world_size
        self.task_num_classes = task_num_classes
        self.labels = labels = getattr(dataset, "labels")
        self.initial_seed = initial_seed
        self.generator: Optional[torch.Generator] = None
        self.current_task_iteration = 0
        self.shuffle = shuffle
        self.debugging = debugging

        if labels is None:
            raise MisconfigurationException(f"Provided {self.dataset} should have an attribute labels.")

        if len(labels) != len(dataset):
            raise MisconfigurationException("Found provided ``labels`` don't match the dataset length.")

        if ((isinstance(global_rank, int) and world_size is None)
            or (isinstance(world_size, int) and global_rank is None)):  # noqa E129
            raise MisconfigurationException("Both ``world_size`` and ``global_rank`` should be provided !")

        self.unique_labels = np.unique(self.labels)

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size is not None and self.world_size > 1

    def set_seed(self, shared: bool = False):
        if shared:
            seed = self.initial_seed
            np_seed = _generate_state(self.initial_seed, 0)
        else:
            seed = self.initial_seed + self.worker_id + self.global_rank + self.current_task_iteration
            np_seed = _generate_state(self.initial_seed, self.worker_id + self.global_rank)

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(np_seed)

    def sample_task_indices(self):
        self.set_seed(shared=True)
        self.selected_indexes = np.random.choice(self.unique_labels, self.task_num_classes, replace=False)
        self.selected_indexes.sort()
        self.task_indices = np.arange(len(self.dataset))[np.in1d(self.labels, self.selected_indexes)]
        self.task_length = len(self.task_indices)
        self.set_seed(shared=False)

    @property
    def worker_rank(self) -> int:
        worker_id = self.worker_id
        is_global_zero = self.global_rank == 0
        return self.global_rank + worker_id + int(not is_global_zero)

    def create_sampler(self):
        data = range(self.task_length)
        if self.world_size == 1 and self.num_workers in (0, 1):
            if self.shuffle:
                self.sampler = RandomSampler(data, generator=self.generator)
            else:
                self.sampler = SequentialSampler(data)
        else:
            num_workers = 1 if self.num_workers in (None, 0) else self.num_workers
            num_replicas = num_workers * self.world_size
            self.sampler = DistributedSampler(
                data, num_replicas=num_replicas, rank=self.worker_rank, shuffle=self.shuffle
            )

            print("INDICES", self.worker_rank, list(self.sampler))

    def __iter__(self):
        if self.generator is None:
            self.generator = torch.Generator().manual_seed(self.initial_seed)
        self.sample_task_indices()
        self.create_sampler()
        self.batch_sampler = BatchSampler(self.sampler, batch_size=self.batch_size, drop_last=self.drop_last)
        self.iter_sampler = iter(self.batch_sampler)
        self.is_first_batch = True
        self.current_task_iteration += 1
        return self

    def __next__(self):
        is_first_batch = self.is_first_batch if self.debugging else (self.is_first_batch and self.worker_id == 0)
        if is_first_batch:
            self.is_first_batch = False
            return {"task_length": len(self.task_indices), "selected_indexes": self.selected_indexes}
        random_indices = next(self.iter_sampler)
        task_indices = [self.task_indices[idx] for idx in random_indices]
        return default_collate([self.dataset[idx] for idx in task_indices])


class ClassificationDataset(Dataset):

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)

    def __getitem__(self, index):
        return (self.inputs[index], self.labels[index])

    def __len__(self):
        return len(self.inputs)


def _test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset(rank, worldsize):
    if worldsize > 1:
        _setup_ddp(rank, worldsize)

    def all_gather(tensor, world_size):
        tensor_list = [torch.zeros_like(tensor, dtype=torch.int64) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, tensor)
        return tensor_list

    initial_seed = seed_everything(42)

    generator = torch.Generator()
    generator.manual_seed(initial_seed)

    num_workers = 2
    batch_size = 4
    dataset_length = 60
    num_classes = 10

    dataset = ClassificationDataset(range(dataset_length), np.random.randint(0, num_classes, dataset_length))
    dataset = MetaLearningDataset(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        global_rank=rank,
        world_size=worldsize,
        initial_seed=initial_seed,
        debugging=True,
        shuffle=True,
    )

    dataset = CaptureIterativeDataset(
        dataset, num_workers=num_workers, batch_size=batch_size, is_inside_workers=True, initial_seed=initial_seed
    )
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=1, generator=generator)

    epoch_results = []
    for _ in range(2):
        iter_dataloader = iter(dataloader)
        batches = []
        while True:
            try:
                batches.append(next(iter_dataloader))
            except StopIteration:
                break
        epoch_results.append(batches)

    assert len(epoch_results) == 2

    assert len(epoch_results[0]) == math.ceil((dataset_length / (num_workers * worldsize)) / batch_size) + 2

    if worldsize == 1:
        assert epoch_results[0][0]["data"]["task_length"] == epoch_results[0][1]["data"]["task_length"]
        assert torch.equal(
            epoch_results[0][0]["data"]["selected_indexes"], epoch_results[0][1]["data"]["selected_indexes"]
        )
        assert epoch_results[0][2][AutoRestartBatchKeys.PL_SAMPLERS]["id"] == 0
        assert epoch_results[0][3][AutoRestartBatchKeys.PL_SAMPLERS]["id"] == 1
        assert not torch.equal(epoch_results[0][2]["data"][0], epoch_results[0][3]["data"][0])
    else:
        first_task_metadata = all_gather(epoch_results[0][0]["data"]["task_length"], worldsize)
        second_task_metadata = all_gather(epoch_results[0][1]["data"]["task_length"], worldsize)
        assert torch.equal(first_task_metadata[0], first_task_metadata[1])
        assert torch.equal(second_task_metadata[0], second_task_metadata[1])
        assert torch.equal(first_task_metadata[0], second_task_metadata[1])

        first_batch_list = all_gather(epoch_results[0][2]["data"][0], worldsize)
        assert not torch.equal(first_batch_list[0], first_batch_list[1])
        second_batch_list = all_gather(epoch_results[0][3]["data"][0], worldsize)
        assert not torch.equal(second_batch_list[0], second_batch_list[1])

    # restarting on epoch 0 / real batch 2
    state_dict = {0: {'iter_sampler': {}}}
    for batch in epoch_results[0][3:5]:
        metadata = parse_metadata(batch)
        for k, v in metadata.items():
            state_dict[0][k].update(v)

    dataset = ClassificationDataset(range(dataset_length), np.random.randint(0, num_classes, dataset_length))
    dataset = MetaLearningDataset(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        global_rank=rank,
        world_size=worldsize,
        initial_seed=initial_seed,
        debugging=True,
        shuffle=True,
    )

    dataset = CaptureIterativeDataset(
        dataset, num_workers=num_workers, batch_size=batch_size, is_inside_workers=True, initial_seed=initial_seed
    )
    dataset.setup(state_dict[0])
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=1, generator=generator)

    epoch_results_restart = []
    for _ in range(2):
        iter_dataloader = iter(dataloader)
        batches = []
        while True:
            try:
                batches.append(next(iter_dataloader))
            except StopIteration:
                break
        epoch_results_restart.append(batches)

    breakpoint()


def test_fast_forward_sampler_iterative_dataset():
    _test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset(0, 1)


@RunIf(skip_windows=True)
def test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset():
    """Make sure result logging works with DDP"""
    tutils.set_random_master_port()
    worldsize = 2
    mp.spawn(
        _test_fast_forward_sampler_with_distributed_sampler_and_iterative_dataset, args=(worldsize, ), nprocs=worldsize
    )
