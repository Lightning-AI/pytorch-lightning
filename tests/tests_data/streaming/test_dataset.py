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
from lightning.data.streaming.shuffle import MinShuffle, NoShuffle
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


def test_streaming_dataset_distributed_min_shuffle(tmpdir):
    seed_everything(42)

    cache = Cache(tmpdir, chunk_size=10)
    for i in range(101):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=True)

    assert isinstance(dataset.shuffle, MinShuffle)

    for i in range(101):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0)
    assert len(dataset) == 41
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 41
    process_1_1 = list(dataset_iter)
    assert len(process_1_1) == 41
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50
    process_1_2 = list(dataset_iter)
    assert len(process_1_2) == 50

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir)
    dataset.distributed_env = _DistributedEnv(2, 1)
    assert len(dataset) == 41
    dataset_iter = iter(dataset)
    process_2_1 = list(dataset_iter)
    assert len(process_2_1) == 41
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50
    process_2_2 = list(dataset_iter)
    assert len(process_2_2) == 50

    assert len([i for i in process_1_1 if i in process_2_1]) == 0
    assert len([i for i in process_1_2 if i in process_2_2]) == 0


def test_streaming_dataset_distributed_no_shuffle(tmpdir):
    seed_everything(42)

    cache = Cache(tmpdir, chunk_size=10)
    for i in range(101):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=False)

    assert isinstance(dataset.shuffle, NoShuffle)

    for i in range(101):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0)
    assert len(dataset) == 50
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50
    process_1_1 = list(dataset_iter)
    assert len(process_1_1) == 50
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50
    process_1_2 = list(dataset_iter)
    assert len(process_1_2) == 50

    dataset = StreamingDataset(name="choco", cache_dir=tmpdir, shuffle=False)
    dataset.distributed_env = _DistributedEnv(2, 1)
    assert len(dataset) == 50
    dataset_iter = iter(dataset)
    process_2_1 = list(dataset_iter)
    assert len(process_2_1) == 50
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50
    process_2_2 = list(dataset_iter)
    assert len(process_2_2) == 50

    _, intervals_per_process = dataset.shuffle.get_chunks_and_intervals_per_process(
        dataset.distributed_env, dataset.current_epoch
    )

    assert process_1_1 == process_1_2

    found_list = []
    for i in process_1_1:
        found = False
        for interval in intervals_per_process[0]:
            if interval[0] <= i <= interval[1]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    found_list = []
    for i in process_2_1:
        found = False
        for interval in intervals_per_process[1]:
            if interval[0] <= i <= interval[1]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    assert len([i for i in process_1_1 if i in process_2_1]) == 0
    assert len([i for i in process_1_2 if i in process_2_2]) == 0
