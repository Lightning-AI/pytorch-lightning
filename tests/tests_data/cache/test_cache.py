import io
import os
from functools import partial

import numpy as np
import pytest
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import Dataset

from lightning import seed_everything
from lightning.data.cache import Cache, CacheDataLoader
from lightning.data.datasets.env import _DistributedEnv
from lightning.fabric import Fabric

_PIL_AVAILABLE = RequirementCache("PIL")
_TORCH_VISION_AVAILABLE = RequirementCache("torchvision")


class ImageDataset(Dataset):
    def __init__(self, tmpdir, cache, size, num_classes, use_transform: bool = False):
        from PIL import Image
        from torchvision import transforms as T

        self.data = []
        self.cache = cache

        seed_everything(42)

        for i in range(size):
            path = os.path.join(tmpdir, f"img{i}.jpeg")
            np_data = np.random.randint(255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(np_data).convert("L")
            img.save(path, format="jpeg", quality=100)
            # read bytes from the file
            with open(path, "rb") as f:
                data = f.read()
            self.data.append({"image": data, "class": np.random.randint(num_classes)})

        self.use_transform = use_transform
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache.filled:
            data = self.cache[index]
            if self.use_transform:
                data["image"] = self.transform(data["image"]).unsqueeze(0)
            return data
        self.cache[index] = {**self.data[index], "index": index}
        return None


def cache_for_image_dataset(num_workers, tmpdir, fabric=None):
    from PIL import Image

    dataset_size = 85

    cache_dir = os.path.join(tmpdir, "cache")
    distributed_env = _DistributedEnv.detect()

    cache = Cache(cache_dir, data_format={"image": "jpeg", "class": "int", "index": "int"}, chunk_size=2 << 12)
    dataset = ImageDataset(tmpdir, cache, dataset_size, 10)
    dataloader = CacheDataLoader(dataset, num_workers=num_workers, batch_size=4)
    dataloader_iter = iter(dataloader)

    for _ in dataloader_iter:
        pass

    for i in range(len(dataset)):
        cached_data = dataset[i]
        original_data = dataset.data[i]
        assert cached_data["class"] == original_data["class"]
        original_image = Image.open(io.BytesIO(original_data["image"]))
        assert cached_data["image"] == original_image

    dataset.use_transform = True

    if distributed_env.world_size == 1:
        indexes = []
        for batch in CacheDataLoader(dataset, num_workers=num_workers, batch_size=4):
            indexes.extend(batch["index"].numpy().tolist())

        assert len(indexes) == dataset_size

    seed_everything(42)

    dataloader = CacheDataLoader(dataset, num_workers=num_workers, batch_size=4, shuffle=True)
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


@pytest.mark.skipif(
    condition=not _PIL_AVAILABLE or not _TORCH_VISION_AVAILABLE, reason="Requires: ['pil', 'torchvision']"
)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_cache_for_image_dataset(num_workers, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir)

    cache_for_image_dataset(num_workers, tmpdir)


def fabric_cache_for_image_dataset(fabric, num_workers, tmpdir):
    cache_for_image_dataset(num_workers, tmpdir, fabric=fabric)


@pytest.mark.skipif(
    condition=not _PIL_AVAILABLE or not _TORCH_VISION_AVAILABLE, reason="Requires: ['pil', 'torchvision']"
)
@pytest.mark.parametrize("num_workers", [2])
def test_cache_for_image_dataset_distributed(num_workers, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir)

    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    fabric.launch(partial(fabric_cache_for_image_dataset, num_workers=num_workers, tmpdir=tmpdir))
