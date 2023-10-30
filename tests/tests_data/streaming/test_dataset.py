# Copyright The Lightning AI team.
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

import pytest
import torch
from lightning import seed_everything
from lightning.data.datasets.env import _DistributedEnv
from lightning.data.streaming import Cache
from lightning.data.streaming import cache as cache_module
from lightning.data.streaming.dataloader import StreamingDataLoader
from lightning.data.streaming.dataset import StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from lightning.data.streaming.shuffle import FullShuffle, NoShuffle
from lightning.pytorch.demos.boring_classes import RandomDataset
from torch.utils.data import DataLoader


def test_streaming_dataset(tmpdir, monkeypatch):
    seed_everything(42)

    os.makedirs(os.path.join(tmpdir, "remote_dir"), exist_ok=True)
    monkeypatch.setattr(cache_module, "_try_create_cache_dir", lambda name: tmpdir)

    with pytest.raises(ValueError, match="The provided dataset `choco` isn't filled up."):
        dataset = StreamingDataset(name="choco", cache_dir=tmpdir)

    dataset = RandomDataset(128, 64)
    dataloader = StreamingDataLoader(dataset, cache_dir=tmpdir, chunk_bytes=2 << 12)
    for batch in dataloader:
        assert isinstance(batch, torch.Tensor)

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, item_loader=TokensLoader(block_size=10))

    assert len(dataset) == 816
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 816

    dataloader = DataLoader(dataset, num_workers=2, batch_size=2)
    assert len(dataloader) == 408


@pytest.mark.parametrize("drop_last", [False, True])
def test_streaming_dataset_distributed_no_shuffle(drop_last, tmpdir):
    seed_everything(42)

    cache = Cache(tmpdir, chunk_size=10)
    for i in range(101):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=False, drop_last=drop_last)

    assert isinstance(dataset.shuffle, NoShuffle)

    for i in range(101):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(1, 0)
    assert len(dataset) == 101

    dataset.distributed_env = _DistributedEnv(2, 0)
    assert len(dataset) == 50 + int(not drop_last)
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50 + int(not drop_last)
    process_1_1 = list(dataset_iter)
    assert len(process_1_1) == 50 + int(not drop_last)
    assert process_1_1[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50 + int(not drop_last)
    process_1_2 = list(dataset_iter)
    assert process_1_2[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert len(process_1_2) == 50 + int(not drop_last)

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=False, drop_last=drop_last)
    dataset.distributed_env = _DistributedEnv(2, 1)
    assert len(dataset) == 50
    dataset_iter = iter(dataset)
    process_2_1 = list(dataset_iter)
    assert process_2_1[:10] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert len(process_2_1) == 50
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50
    process_2_2 = list(dataset_iter)
    assert process_2_2[:10] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    assert len(process_2_2) == 50

    _, intervals_per_ranks = dataset.shuffle.get_chunks_and_intervals_per_ranks(
        dataset.distributed_env, dataset.current_epoch
    )

    assert process_1_1 == process_1_2

    found_list = []
    for i in process_1_1:
        found = False
        for interval in intervals_per_ranks[0]:
            if interval[0] <= i <= interval[1]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    found_list = []
    for i in process_2_1:
        found = False
        for interval in intervals_per_ranks[1]:
            if interval[0] <= i <= interval[1]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    assert len([i for i in process_1_1 if i in process_2_1]) == 0
    assert len([i for i in process_1_2 if i in process_2_2]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
def test_streaming_dataset_distributed_full_shuffle_odd(drop_last, tmpdir):
    seed_everything(42)

    cache = Cache(tmpdir, chunk_size=10)
    for i in range(1097):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=True, drop_last=drop_last)

    assert isinstance(dataset.shuffle, FullShuffle)

    for i in range(1097):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0)
    assert len(dataset) == 548
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 548
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [785, 788, 782, 783, 789, 787, 786, 781, 784, 780]
    assert len(process_1_1) == 548

    dataset_2 = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=True, drop_last=drop_last)
    assert isinstance(dataset_2.shuffle, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(2, 1)
    assert len(dataset_2) == 548 + int(not drop_last)
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 548 + int(not drop_last)
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [939, 938, 252, 259, 257, 255, 258, 253, 250, 251]
    assert len(process_2_1) == 548 + int(not drop_last)
    assert len([i for i in process_1_1 if i in process_2_1]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
def test_streaming_dataset_distributed_full_shuffle_even(drop_last, tmpdir):
    seed_everything(42)

    cache = Cache(tmpdir, chunk_size=10)
    for i in range(1222):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=True, drop_last=drop_last)

    assert isinstance(dataset.shuffle, FullShuffle)

    for i in range(1222):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0)
    assert len(dataset) == 611
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 611
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [185, 184, 182, 189, 187, 181, 183, 180, 186, 188]
    assert len(process_1_1) == 611

    dataset_2 = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=True, drop_last=drop_last)
    assert isinstance(dataset_2.shuffle, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(2, 1)
    assert len(dataset_2) == 611
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 611
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [813, 815, 816, 812, 818, 811, 817, 814, 819, 277]
    assert len(process_2_1) == 611

    assert len([i for i in process_1_1 if i in process_2_1]) == 0


def test_streaming_dataset_deepcopy(tmpdir, monkeypatch):
    seed_everything(42)

    remote_dir = os.path.join(tmpdir, "remote_dir")

    os.makedirs(remote_dir, exist_ok=True)

    cache = Cache(remote_dir, chunk_size=10)
    for i in range(10):
        cache[i] = i

    cache.done()
    cache.merge()

    monkeypatch.setattr(cache_module, "_find_remote_dir", lambda x, y: (str(remote_dir), True))

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=True)
    assert dataset.cache._reader._prepare_thread is None
    _ = dataset[0]
    assert dataset.cache._reader._prepare_thread
    dataloader = DataLoader(dataset, num_workers=1)

    batches = []
    for batch in dataloader:
        batches.append(batch)

    assert len(batches) == 10
