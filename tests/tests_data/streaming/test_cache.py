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
import json
import os
import sys
from functools import partial

import numpy as np
import pytest
import torch
from lightning import seed_everything
from lightning.data.streaming import Cache
from lightning.data.streaming.dataloader import StreamingDataLoader
from lightning.data.streaming.dataset import StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from lightning.data.streaming.serializers import Serializer
from lightning.data.utilities.env import _DistributedEnv
from lightning.fabric import Fabric
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.test.warning import no_warning_call
from torch.utils.data import Dataset

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

    streaming_dataset = StreamingDataset(input_dir=cache_dir)
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

    # you encode data
    for i in range(100):
        cache[i] = i

    # I am done, write the index ...
    cache.done()
    cache.merge()

    # please, decode the data for me.
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


def test_create_oversized_chunk_single_item(tmp_path):
    cache = Cache(str(tmp_path), chunk_bytes=700)
    with pytest.warns(UserWarning, match="An item was larger than the target chunk size"):
        cache[0] = np.random.randint(0, 10, size=(10000,), dtype=np.uint8)


def test_create_undersized_and_oversized_chunk(tmp_path):
    cache = Cache(str(tmp_path), chunk_bytes=9000)  # target: 9KB chunks
    with no_warning_call(UserWarning):
        cache[0] = np.random.randint(0, 10, size=(500,), dtype=np.uint8)  # will result in undersized chunk
        cache[1] = np.random.randint(0, 10, size=(10000,), dtype=np.uint8)  # will result in oversized chunk
    with pytest.warns(UserWarning, match="An item was larger than the target chunk size"):
        cache[2] = np.random.randint(0, 10, size=(150,), dtype=np.uint8)
    with no_warning_call(UserWarning):
        cache[3] = np.random.randint(0, 10, size=(200,), dtype=np.uint8)

    cache.done()
    cache.merge()

    assert len(os.listdir(tmp_path)) == 4  # 3 chunks + 1 index file
    with open(tmp_path / "index.json") as file:
        index = json.load(file)

    chunks = index["chunks"]
    assert chunks[0]["chunk_size"] == 1
    assert chunks[0]["filename"] == "chunk-0-0.bin"
    assert chunks[1]["chunk_size"] == 1
    assert chunks[1]["filename"] == "chunk-0-1.bin"
    assert chunks[2]["chunk_size"] == 2
    assert chunks[2]["filename"] == "chunk-0-2.bin"


class CustomData:
    pass


class CustomSerializer(Serializer):
    def serialize(self, data):
        return np.array([1]).tobytes(), None

    def deserialize(self, data: bytes):
        return data

    def can_serialize(self, data) -> bool:
        return isinstance(data, CustomData)


def test_custom_serializer(tmpdir):
    cache = Cache(input_dir=str(tmpdir), serializers={"custom": CustomSerializer()}, chunk_size=1)
    for i in range(10):
        cache[i] = (CustomData(),)
    cache.done()
    cache.merge()
    assert isinstance(cache[0][0], bytes)


def test_cache_for_text_tokens(tmpdir):
    seed_everything(42)

    block_size = 1024 + 1
    cache = Cache(input_dir=str(tmpdir), chunk_size=block_size * 11, item_loader=TokensLoader(block_size))
    text_idxs_list = []

    counter = 0
    while True:
        text_ids = torch.randint(0, 1000, (np.random.randint(0, 1000),)).to(torch.int)
        text_idxs_list.append(text_ids)
        chunk_filepath = cache._add_item(counter, text_ids)
        if chunk_filepath:
            break
        counter += 1

    cache.done()
    cache.merge()

    assert len(cache) == 10

    cache_0 = cache[0]
    cache_1 = cache[1]
    assert len(cache_0) == block_size
    assert len(cache_1) == block_size
    assert not torch.equal(cache_0, cache[1])
    indices = torch.cat(text_idxs_list, dim=0)
    assert torch.equal(cache_0, indices[: len(cache_0)])
    assert torch.equal(cache_1, indices[len(cache_0) : len(cache_0) + len(cache_1)])

    with pytest.raises(ValueError, match="TokensLoader"):
        len(Cache(str(tmpdir), chunk_size=block_size * 11))
