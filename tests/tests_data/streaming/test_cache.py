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
import sys
from functools import partial

import numpy as np
import pytest
import torch
from lightning import seed_everything
from lightning.data.datasets.env import _DistributedEnv
from lightning.data.streaming import Cache
from lightning.data.streaming import cache as cache_module
from lightning.data.streaming.dataloader import StreamingDataLoader
from lightning.data.streaming.dataset import StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from lightning.fabric import Fabric
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader, Dataset

_PIL_AVAILABLE = RequirementCache("PIL")
_TORCH_VISION_AVAILABLE = RequirementCache("torchvision")


class ImageDataset(Dataset):
    def __init__(self, tmpdir, cache, size, num_classes):
        from PIL import Image

        self.data = []
        self.cache = cache

        seed_everything(42)

        for i in range(size):
            path = os.path.join(tmpdir, f"img{i}.jpeg")
            np_data = np.random.randint(255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(np_data).convert("L")
            img.save(path, format="jpeg", quality=100)
            self.data.append({"image": path, "class": np.random.randint(num_classes)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache.filled:
            return self.cache[index]
        self.cache[index] = {**self.data[index], "index": index}
        return None


def _cache_for_image_dataset(num_workers, tmpdir, fabric=None):
    from PIL import Image
    from torchvision.transforms import PILToTensor

    dataset_size = 85

    cache_dir = os.path.join(tmpdir, "cache")
    distributed_env = _DistributedEnv.detect()

    cache = Cache(cache_dir, chunk_size=10)
    dataset = ImageDataset(tmpdir, cache, dataset_size, 10)
    dataloader = StreamingDataLoader(dataset, num_workers=num_workers, batch_size=4)

    for _ in dataloader:
        pass

    # Not strictly required but added to avoid race condition
    if distributed_env.world_size > 1:
        fabric.barrier()

    assert cache.filled

    for i in range(len(dataset)):
        cached_data = dataset[i]
        original_data = dataset.data[i]
        assert cached_data["class"] == original_data["class"]
        original_array = PILToTensor()(Image.open(original_data["image"]))
        assert torch.equal(original_array, cached_data["image"])

    if distributed_env.world_size == 1:
        indexes = []
        dataloader = StreamingDataLoader(dataset, num_workers=num_workers, batch_size=4)
        for batch in dataloader:
            if batch:
                indexes.extend(batch["index"].numpy().tolist())
        assert len(indexes) == dataset_size

    seed_everything(42)

    dataloader = StreamingDataLoader(dataset, num_workers=num_workers, batch_size=4, shuffle=True)
    dataloader_iter = iter(dataloader)

    indexes = []
    for batch in dataloader_iter:
        indexes.extend(batch["index"].numpy().tolist())

    if distributed_env.world_size == 1:
        assert len(indexes) == dataset_size

    indexes2 = []
    for batch in dataloader_iter:
        indexes2.extend(batch["index"].numpy().tolist())

    assert indexes2 != indexes

    streaming_dataset = StreamingDataset(name="dummy", cache_dir=cache_dir)
    for i in range(len(streaming_dataset)):
        cached_data = streaming_dataset[i]
        original_data = dataset.data[i]
        assert cached_data["class"] == original_data["class"]
        original_array = PILToTensor()(Image.open(original_data["image"]))
        assert torch.equal(original_array, cached_data["image"])

    streaming_dataset_iter = iter(streaming_dataset)
    for _ in streaming_dataset_iter:
        pass


@pytest.mark.skipif(
    condition=not _PIL_AVAILABLE or not _TORCH_VISION_AVAILABLE, reason="Requires: ['pil', 'torchvision']"
)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_cache_for_image_dataset(num_workers, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir)

    _cache_for_image_dataset(num_workers, tmpdir)


def _fabric_cache_for_image_dataset(fabric, num_workers, tmpdir):
    _cache_for_image_dataset(num_workers, tmpdir, fabric=fabric)


@pytest.mark.skipif(
    condition=not _PIL_AVAILABLE or not _TORCH_VISION_AVAILABLE or sys.platform == "win32",
    reason="Requires: ['pil', 'torchvision']",
)
@pytest.mark.parametrize("num_workers", [2])
def test_cache_for_image_dataset_distributed(num_workers, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir)

    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    fabric.launch(partial(_fabric_cache_for_image_dataset, num_workers=num_workers, tmpdir=tmpdir))


def test_cache_with_simple_format(tmpdir):
    cache_dir = os.path.join(tmpdir, "cache1")
    os.makedirs(cache_dir)

    cache = Cache(cache_dir, chunk_bytes=90)

    for i in range(100):
        cache[i] = i

    cache.done()
    cache.merge()

    for i in range(100):
        assert i == cache[i]

    cache_dir = os.path.join(tmpdir, "cache2")
    os.makedirs(cache_dir)

    cache = Cache(cache_dir, chunk_bytes=90)

    for i in range(100):
        cache[i] = [i, {0: [i + 1]}]

    cache.done()
    cache.merge()

    for i in range(100):
        assert [i, {0: [i + 1]}] == cache[i]


def test_cache_with_auto_wrapping(tmpdir):
    os.makedirs(os.path.join(tmpdir, "cache_1"), exist_ok=True)

    dataset = RandomDataset(64, 64)
    dataloader = StreamingDataLoader(dataset, cache_dir=os.path.join(tmpdir, "cache_1"), chunk_bytes=2 << 12)
    for batch in dataloader:
        assert isinstance(batch, torch.Tensor)
    assert sorted(os.listdir(os.path.join(tmpdir, "cache_1"))) == [
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "index.json",
    ]
    # Your dataset is optimised for the cloud

    class RandomDatasetAtRuntime(Dataset):
        def __init__(self, size: int, length: int):
            self.len = length
            self.size = size

        def __getitem__(self, index: int) -> torch.Tensor:
            return torch.randn(1, self.size)

        def __len__(self) -> int:
            return self.len

    os.makedirs(os.path.join(tmpdir, "cache_2"), exist_ok=True)
    dataset = RandomDatasetAtRuntime(64, 64)
    dataloader = StreamingDataLoader(dataset, cache_dir=os.path.join(tmpdir, "cache_2"), chunk_bytes=2 << 12)
    with pytest.raises(ValueError, match="Your dataset items aren't deterministic"):
        for batch in dataloader:
            pass


def test_cache_with_name(tmpdir, monkeypatch):
    with pytest.raises(FileNotFoundError, match="The provided cache directory"):
        Cache(name="something")

    os.makedirs(os.path.join(tmpdir, "something"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "remote_dir"), exist_ok=True)
    monkeypatch.setattr(cache_module, "_try_create_cache_dir", lambda name: os.path.join(tmpdir, name))

    monkeypatch.setattr(
        cache_module, "_find_remote_dir", lambda name, _: (os.path.join(tmpdir, "remote_dir", "version_0"), True)
    )
    cache = Cache(name="something")
    assert cache._writer._chunk_size == 2
    assert cache._writer._cache_dir == os.path.join(tmpdir, "something", "version_0")
    assert cache._reader._remote_dir == os.path.join(tmpdir, "remote_dir", "version_0")


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
