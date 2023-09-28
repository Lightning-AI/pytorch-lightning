import os

import numpy as np
import pytest
from lightning import seed_everything
from lightning.data.cache import Cache, CacheDataLoader
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import Dataset

_PIL_AVAILABLE = RequirementCache("PIL")


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
            # read bytes from the file
            with open(path, "rb") as f:
                data = f.read()
            self.data.append({"image": data, "class": np.random.randint(num_classes)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache.filled:
            return self.cache[index]
        self.cache[index] = self.data[index]
        return None


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_cache_for_image_dataset(num_workers, tmpdir):
    import io

    from PIL import Image

    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir)
    cache = Cache(cache_dir, data_format={"image": "jpeg", "class": "int"}, chunk_size=2 << 12)
    dataset = ImageDataset(tmpdir, cache, 85, 10)
    for _ in CacheDataLoader(dataset, num_workers=num_workers, batch_size=4):
        pass

    for i in range(len(dataset)):
        cached_data = dataset[i]
        original_data = dataset.data[i]
        assert cached_data["class"] == original_data["class"]
        original_image = Image.open(io.BytesIO(original_data["image"]))
        assert cached_data["image"] == original_image
