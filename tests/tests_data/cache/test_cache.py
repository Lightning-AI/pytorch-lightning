import os

import numpy as np
import pytest
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import Dataset

from lightning import seed_everything
from lightning.data.cache import Cache, CacheDataLoader

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


@pytest.mark.skipif(
    condition=not _PIL_AVAILABLE or not _TORCH_VISION_AVAILABLE, reason="Requires: ['pil', 'torchvision']"
)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_cache_for_image_dataset(num_workers, tmpdir):
    import io

    from PIL import Image

    dataset_size = 85

    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir)
    cache = Cache(cache_dir, data_format={"image": "jpeg", "class": "int", "index": "int"}, chunk_size=2 << 12)
    dataset = ImageDataset(tmpdir, cache, dataset_size, 10)
    for _ in CacheDataLoader(dataset, num_workers=num_workers, batch_size=4):
        pass

    for i in range(len(dataset)):
        cached_data = dataset[i]
        original_data = dataset.data[i]
        assert cached_data["class"] == original_data["class"]
        original_image = Image.open(io.BytesIO(original_data["image"]))
        assert cached_data["image"] == original_image

    dataset.use_transform = True

    indexes = []
    for batch in CacheDataLoader(dataset, num_workers=num_workers, batch_size=4):
        indexes.extend(batch["index"].numpy().tolist())

    assert indexes == list(range(dataset_size))

    seed_everything(42)

    dataloader = CacheDataLoader(dataset, num_workers=num_workers, batch_size=4, shuffle=True)

    indexes = []
    for batch in dataloader:
        indexes.extend(batch["index"].numpy().tolist())

    assert len(indexes) == dataset_size

    indexes2 = []
    for batch in dataloader:
        indexes2.extend(batch["index"].numpy().tolist())

    assert indexes2 != indexes
