import os
from typing import Any, List

import numpy as np
import pytest
from lightning.data.cache.dataset_optimizer import DatasetOptimizer
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


class TestDatasetOptimizer(DatasetOptimizer):
    def prepare_dataset_structure(self, src_dir: str, filepaths: List[str]) -> List[Any]:
        assert len(filepaths) == 100
        return filepaths


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_data_optimizer(tmpdir, monkeypatch):
    from PIL import Image

    imgs = []
    for i in range(20):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(tmpdir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache")
    monkeypatch.setenv("HOME_FOLDER", home_dir)
    monkeypatch.setenv("CACHE_FOLDER", cache_dir)
    datasetOptimizer = TestDatasetOptimizer(
        name="dummy_dataset", src_dir=tmpdir, chunk_size=2, num_workers=2, num_downloaders=1, worker_type="process"
    )
    datasetOptimizer.run()

    assert sorted(os.listdir(cache_dir)) == ["data", "dummy_dataset"]
    assert sorted(os.listdir(os.path.join(cache_dir, "dummy_dataset"))) == [
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-0-2.bin",
        "chunk-0-3.bin",
        "chunk-0-4.bin",
        "chunk-1-0.bin",
        "chunk-1-1.bin",
        "chunk-1-2.bin",
        "chunk-1-3.bin",
        "chunk-1-4.bin",
        "index.json",
    ]

    breakpoint()
