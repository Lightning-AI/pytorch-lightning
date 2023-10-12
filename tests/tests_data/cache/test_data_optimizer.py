import os
from typing import Any, List

import numpy as np
import pytest
from lightning.data.cache.dataset_optimizer import DatasetOptimizer
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


class TestDatasetOptimizer(DatasetOptimizer):
    def prepare_dataset_structure(self, src_dir: str, filepaths: List[str]) -> List[Any]:
        assert len(filepaths) == 30
        return filepaths


@pytest.mark.parametrize("delete_cached_files", [False, True])
@pytest.mark.parametrize("fast_dev_run", [False, True])
@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_data_optimizer(fast_dev_run, delete_cached_files, tmpdir, monkeypatch):
    from PIL import Image

    imgs = []
    for i in range(30):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(tmpdir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache")
    monkeypatch.setenv("HOME_FOLDER", home_dir)
    monkeypatch.setenv("CACHE_FOLDER", cache_dir)
    datasetOptimizer = TestDatasetOptimizer(
        name="dummy_dataset",
        src_dir=tmpdir,
        chunk_size=2,
        num_workers=2,
        num_downloaders=1,
        remote_src_dir=tmpdir,
        worker_type="process",
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
    )
    datasetOptimizer.run()

    assert sorted(os.listdir(cache_dir)) == ["data", "dummy_dataset"]

    fast_dev_run_enabled_chunks = [
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

    fast_dev_run_disabled_chunks = [
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-0-2.bin",
        "chunk-0-3.bin",
        "chunk-0-4.bin",
        "chunk-0-5.bin",
        "chunk-0-6.bin",
        "chunk-0-7.bin",
        "chunk-1-0.bin",
        "chunk-1-1.bin",
        "chunk-1-2.bin",
        "chunk-1-3.bin",
        "chunk-1-4.bin",
        "chunk-1-5.bin",
        "chunk-1-6.bin",
        "chunk-1-7.bin",
        "index.json",
    ]

    chunks = fast_dev_run_enabled_chunks if fast_dev_run else fast_dev_run_disabled_chunks

    assert sorted(os.listdir(os.path.join(cache_dir, "dummy_dataset"))) == chunks

    files = []
    for _, _, filenames in os.walk(os.path.join(cache_dir, "data")):
        files.extend(filenames)

    expected = (0 if delete_cached_files else 20) if fast_dev_run else (0 if delete_cached_files else 30)
    assert len(files) == expected


@pytest.mark.parametrize("delete_cached_files", [False])
@pytest.mark.parametrize("fast_dev_run", [False])
@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_data_optimizer_distributed(fast_dev_run, delete_cached_files, tmpdir, monkeypatch):
    from PIL import Image

    imgs = []
    for i in range(30):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(tmpdir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    monkeypatch.setenv("HOME_FOLDER", home_dir)

    remote_dst_dir = os.path.join(tmpdir, "dst")
    os.makedirs(remote_dst_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_1")
    monkeypatch.setenv("CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setenv("NODE_RANK", "0")
    datasetOptimizer = TestDatasetOptimizer(
        name="dummy_dataset",
        src_dir=tmpdir,
        chunk_size=2,
        num_workers=2,
        num_downloaders=1,
        remote_src_dir=tmpdir,
        worker_type="process",
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
        remote_dst_dir=remote_dst_dir,
    )
    datasetOptimizer.run()

    assert sorted(os.listdir(cache_dir)) == ["data", "dummy_dataset"]

    fast_dev_run_disabled_chunks_0 = [
        "0-index.json",
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-0-2.bin",
        "chunk-0-3.bin",
        "chunk-1-0.bin",
        "chunk-1-1.bin",
        "chunk-1-2.bin",
        "chunk-1-3.bin",
    ]

    assert sorted(os.listdir(os.path.join(cache_dir, "dummy_dataset"))) == fast_dev_run_disabled_chunks_0

    cache_dir = os.path.join(tmpdir, "cache_2")
    monkeypatch.setenv("CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setenv("NODE_RANK", "1")
    datasetOptimizer = TestDatasetOptimizer(
        name="dummy_dataset",
        src_dir=tmpdir,
        chunk_size=2,
        num_workers=2,
        num_downloaders=1,
        remote_src_dir=tmpdir,
        worker_type="process",
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
        remote_dst_dir=remote_dst_dir,
    )
    datasetOptimizer.run()

    assert sorted(os.listdir(cache_dir)) == ["data", "dummy_dataset"]

    fast_dev_run_disabled_chunks_1 = [
        "chunk-2-0.bin",
        "chunk-2-1.bin",
        "chunk-2-2.bin",
        "chunk-2-3.bin",
        "chunk-3-0.bin",
        "chunk-3-1.bin",
        "chunk-3-2.bin",
        "chunk-3-3.bin",
        "index.json",
    ]
    assert sorted(os.listdir(os.path.join(cache_dir, "dummy_dataset"))) == fast_dev_run_disabled_chunks_1

    expected = sorted(fast_dev_run_disabled_chunks_0 + fast_dev_run_disabled_chunks_1 + ["1-index.json"])
    assert sorted(os.listdir(remote_dst_dir)) == expected
