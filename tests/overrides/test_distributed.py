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
from collections.abc import Iterable

import pytest
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import _InfiniteConstantSampler, DataLoader
from torch.utils.data.dataset import IterableDataset

from pytorch_lightning import Callback, seed_everything, Trainer
from pytorch_lightning.overrides.distributed import (
    CaptureIterativeDataset,
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


def test_fast_forward_sampler_replacement(tmpdir):

    seed_everything(42)

    class CustomBatchSampler(BatchSampler):
        pass

    class CheckFastForwardSamplerInjection(Callback):

        def __init__(self):
            self.has_called = False

        def on_train_batch_end(self, trainer, *args) -> None:
            sampler = trainer.train_dataloader.loaders.fast_forward_sampler
            wrapping_batch_sampler = isinstance(sampler.sampler, BatchSampler)

            num_batches = 2

            if wrapping_batch_sampler:
                assert isinstance(sampler, FastForwardSampler)
                current_iteration = num_batches
            else:
                assert isinstance(sampler, FastForwardSampler)
                current_iteration = 2 * 3

            if trainer.fit_loop.batch_idx == 1:
                assert sampler.state_dict(num_batches)["current_iteration"] == current_iteration
                self.has_called = True

    dataset = RandomDataset(32, 64)
    train_dataloader = DataLoader(dataset, batch_size=3, num_workers=1)
    trainer_kwargs = dict(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=2, num_sanity_val_steps=0, limit_val_batches=0
    )
    model = BoringModel()
    callback = CheckFastForwardSamplerInjection()
    trainer = Trainer(**trainer_kwargs, callbacks=callback)
    trainer.fit(model, train_dataloader=train_dataloader)

    train_dataloader = DataLoader(
        dataset,
        batch_sampler=CustomBatchSampler(batch_size=8, sampler=SequentialSampler(dataset), drop_last=True),
        num_workers=2,
    )

    trainer = Trainer(**trainer_kwargs, callbacks=callback)
    trainer.fit(model, train_dataloader=train_dataloader)


class IterativeRandomDataset(IterableDataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.sampler = RandomSampler(self.data)

    def __iter__(self):
        self.index = 0
        self.iter_sampler = iter(self.sampler)
        return self

    def __next__(self):
        return self.data[next(self.iter_sampler)]


def test_fast_forward_sampler_iterative_dataset(tmpdir):

    seed_everything(42)

    class CustomBatchSampler(BatchSampler):
        pass

    class CheckFastForwardSamplerInjection(Callback):

        def __init__(self):
            self.has_called = False

        def on_train_batch_end(self, trainer, *args) -> None:
            assert isinstance(trainer.train_dataloader.loaders.sampler, _InfiniteConstantSampler)
            assert isinstance(trainer.train_dataloader.loaders.dataset, CaptureIterativeDataset)
            sampler = trainer.train_dataloader.loaders.dataset.sampler
            assert isinstance(sampler, FastForwardSampler)

    dataset = IterativeRandomDataset(32, 20)
    train_dataloader = DataLoader(dataset, batch_size=3, num_workers=0)
    trainer_kwargs = dict(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=2, num_sanity_val_steps=0, limit_val_batches=0
    )
    model = BoringModel()
    callback = CheckFastForwardSamplerInjection()
    trainer = Trainer(**trainer_kwargs, callbacks=callback)
    trainer.fit(model, train_dataloader=train_dataloader)
