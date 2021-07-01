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
import random
from collections.abc import Iterable
from typing import Optional

import pytest
import torch
from torch.functional import Tensor
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader, get_worker_info
from torch.utils.data.dataset import Dataset

from pytorch_lightning import Callback, seed_everything, Trainer
from pytorch_lightning.overrides.distributed import (
    FastForwardSampler,
    IndexBatchSamplerWrapper,
    UnrepeatedDistributedSampler,
)
from pytorch_lightning.utilities.data import has_len
from tests.helpers.boring_model import BoringModel, RandomDataset


@pytest.mark.parametrize("shuffle", [False, True])
def test_unrepeated_distributed_sampler(shuffle, tmpdir):
    """Test each rank will receive a different number of elements."""

    seed_everything(42)
    world_size = 4
    samplers = []
    dataset = range(103)
    for rank in range(world_size):
        samplers.append(UnrepeatedDistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=shuffle))

    indices = [list(s) for s in samplers]
    assert len(indices[0]) == 26
    assert len(indices[1]) == 26
    assert len(indices[2]) == 26
    assert len(indices[3]) == 25

    assert indices[0][-1] == 18 if shuffle else 100
    assert indices[1][-1] == 30 if shuffle else 101
    assert indices[2][-1] == 29 if shuffle else 102
    assert indices[3][-1] == 35 if shuffle else 99


def test_index_batch_sampler(tmpdir):
    """Test `IndexBatchSampler` properly extracts indices."""
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = IndexBatchSamplerWrapper(batch_sampler)

    assert batch_sampler.batch_size == index_batch_sampler.batch_size
    assert batch_sampler.drop_last == index_batch_sampler.drop_last
    assert batch_sampler.sampler is sampler

    for batch in index_batch_sampler:
        assert index_batch_sampler.batch_indices == batch


def test_index_batch_sampler_methods():
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = IndexBatchSamplerWrapper(batch_sampler)

    assert isinstance(index_batch_sampler, Iterable)
    assert has_len(index_batch_sampler)


def test_fast_forward_on_batch_sampler():
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = FastForwardSampler(batch_sampler)

    assert isinstance(index_batch_sampler, Iterable)
    assert has_len(index_batch_sampler)

    index_batch_sampler_iter = iter(index_batch_sampler)

    assert next(index_batch_sampler_iter) == [0, 1, 2]
    assert next(index_batch_sampler_iter) == [3, 4, 5]

    state_dict = index_batch_sampler.state_dict()
    assert state_dict["current_iteration"] == 2

    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = FastForwardSampler(batch_sampler)
    index_batch_sampler.load_state_dict(state_dict)

    index_batch_sampler_iter = iter(index_batch_sampler)
    assert next(index_batch_sampler_iter) == [6, 7, 8]


def test_fast_forward_on_sequential_sampler():
    dataset = range(15)
    sampler = FastForwardSampler(SequentialSampler(dataset))
    batch_sampler = BatchSampler(sampler, 3, False)

    batch_sampler_iter = iter(batch_sampler)

    assert next(batch_sampler_iter) == [0, 1, 2]
    assert next(batch_sampler_iter) == [3, 4, 5]

    state_dict = sampler.state_dict()
    assert state_dict["current_iteration"] == 6

    dataset = range(15)
    sampler = FastForwardSampler(SequentialSampler(dataset))
    batch_sampler = BatchSampler(sampler, 3, False)
    sampler.load_state_dict(state_dict)

    batch_sampler_iter = iter(batch_sampler)
    assert next(batch_sampler_iter) == [6, 7, 8]


def test_fast_forward_on_random_sampler():
    seed_everything(42)

    dataset = range(15)
    sampler = FastForwardSampler(RandomSampler(dataset))
    batch_sampler = BatchSampler(sampler, 3, False)

    batch_sampler_iter = iter(batch_sampler)

    assert next(batch_sampler_iter) == [14, 9, 1]
    assert next(batch_sampler_iter) == [7, 11, 3]
    assert next(batch_sampler_iter) == [12, 8, 2]

    state_dict = sampler.state_dict()
    assert state_dict["current_iteration"] == 9
    state_dict["current_iteration"] = 6

    dataset = range(15)
    sampler = FastForwardSampler(RandomSampler(dataset))
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
        sampler.load_state_dict(sampler.state_dict())
    assert has_raised


class VerboseRandomDataset(RandomDataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class WrapperDataset(Dataset):

    def __init__(self, dataset, seeds: Optional[Tensor] = None):
        self.dataset = dataset
        self.seeds = seeds

    def __getitem__(self, index):
        worker_info = get_worker_info()
        if self.seeds:
            seed = int(self.seeds[worker_info.id])
            random.seed(seed)
            torch.random.set_rng_state(torch.manual_seed(seed).get_state())
            self.seeds = None
        data = self.dataset[index]
        seed = str(torch.random.seed())
        print("current_seed", worker_info.id, seed)
        return {"data": data, "seed": seed}

    def __len__(self):
        return len(self.dataset)


def test_fast_forward_sampler_replacement(tmpdir):

    seed_everything(42)

    class CustomBatchSampler(BatchSampler):
        pass

    class CheckFastForwardSamplerInjection(Callback):

        def __init__(self, using_batch_sampler: bool = True):
            self.using_batch_sampler = using_batch_sampler

        def on_train_batch_end(self, trainer, *args) -> None:
            if self.using_batch_sampler:
                sampler = trainer.train_dataloader.loaders.batch_sampler
                assert isinstance(sampler, FastForwardSampler)
            else:
                sampler = trainer.train_dataloader.sampler
                assert isinstance(sampler, FastForwardSampler)
                current_iteration = 2 * 3

            if trainer.fit_loop.batch_idx == 2:
                assert sampler.state_dict()["current_iteration"] == current_iteration

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            breakpoint()
            return super().training_step(batch, batch_idx)

    dataset = RandomDataset(32, 64)

    train_dataloader = DataLoader(dataset, batch_size=3)

    trainer_kwargs = dict(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=3, num_sanity_val_steps=0)
    model = TestModel()
    trainer = Trainer(**trainer_kwargs, callbacks=CheckFastForwardSamplerInjection(using_batch_sampler=False))
    trainer.fit(model, train_dataloader=train_dataloader)

    train_dataloader = DataLoader(
        dataset,
        batch_sampler=CustomBatchSampler(batch_size=8, sampler=SequentialSampler(dataset), drop_last=True),
        prefetch_factor=2,
        num_workers=2,
    )


def test_generator(tmpdir):

    class CustomBatchSampler(BatchSampler):
        pass

    dataset = WrapperDataset(VerboseRandomDataset(2, 64))

    generator = torch.Generator()
    initial_state = generator.get_state()

    train_dataloader = DataLoader(
        dataset,
        batch_sampler=CustomBatchSampler(batch_size=8, sampler=SequentialSampler(dataset), drop_last=True),
        prefetch_factor=2,
        num_workers=2,
        generator=generator
    )

    iter_train_dataloader = iter(train_dataloader)
    batches = []
    for _ in range(4):
        batches.append(next(iter_train_dataloader))

    all_seeds = [int(s) for b in batches for s in b["seed"]]

    assert iter_train_dataloader._num_yielded == 4
    base_seed = iter_train_dataloader._base_seed

    seeds = {0: batches[0]["seed"][-1], 1: batches[1]["seed"][-1]}

    dataset = WrapperDataset(VerboseRandomDataset(2, 64), seeds=seeds)

    generator.set_state(initial_state)

    train_dataloader = DataLoader(
        dataset,
        batch_sampler=CustomBatchSampler(batch_size=8, sampler=SequentialSampler(dataset), drop_last=True),
        prefetch_factor=2,
        num_workers=2,
        generator=generator
    )

    iter_train_dataloader = iter(train_dataloader)
    assert base_seed == iter_train_dataloader._base_seed

    batches_restart = []
    for _ in range(2):
        batches_restart.append(next(iter_train_dataloader))

    all_restart_seeds = [int(s) for b in batches_restart for s in b["seed"]]

    [(sr, s) for sr in all_restart_seeds for s in all_seeds]

    for bf, ba in zip(batches[2:], batches_restart):
        assert torch.equal(ba["data"], bf["data"])
