import os
import sys
from typing import Any, List
from unittest import mock

import numpy as np
import pytest
import torch
from lightning import LightningDataModule, seed_everything
from lightning.data.streaming.dataset_optimizer import (
    DatasetOptimizer,
    _download_data_target,
    _remove_target,
    _upload_fn,
)
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_upload_fn(tmpdir):
    src_dir = os.path.join(tmpdir, "src_dir")
    os.makedirs(src_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    remote_dst_dir = os.path.join(tmpdir, "remote_dst_dir")
    os.makedirs(remote_dst_dir, exist_ok=True)

    filepath = os.path.join(src_dir, "a.txt")

    with open(filepath, "w") as f:
        f.write("HERE")

    upload_queue = mock.MagicMock()

    paths = [filepath, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return value

    upload_queue.get = fn

    remove_queue = mock.MagicMock()

    assert os.listdir(remote_dst_dir) == []

    _upload_fn(upload_queue, remove_queue, cache_dir, remote_dst_dir)

    assert os.listdir(remote_dst_dir) == ["a.txt"]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_remove_target(tmpdir):
    src_dir = os.path.join(tmpdir, "src_dir")
    os.makedirs(src_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    filepath = os.path.join(cache_dir, "a.txt")

    with open(filepath, "w") as f:
        f.write("HERE")

    filepath = os.path.join(src_dir, "a.txt")

    queue_in = mock.MagicMock()

    paths = [filepath, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return [value]

    queue_in.get = fn

    assert os.listdir(cache_dir) == ["a.txt"]

    _remove_target(src_dir, cache_dir, queue_in)

    assert os.listdir(cache_dir) == []


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_download_data_target(tmpdir):
    src_dir = os.path.join(tmpdir, "src_dir")
    os.makedirs(src_dir, exist_ok=True)

    remote_src_dir = os.path.join(tmpdir, "remote_src_dir")
    os.makedirs(remote_src_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    filepath = os.path.join(remote_src_dir, "a.txt")

    with open(filepath, "w") as f:
        f.write("HERE")

    filepath = os.path.join(src_dir, "a.txt")

    with open(filepath, "w") as f:
        f.write("HERE")

    queue_in = mock.MagicMock()

    paths = [filepath, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return (0, [value])

    queue_in.get = fn

    queue_out = mock.MagicMock()
    _download_data_target(src_dir, remote_src_dir, cache_dir, queue_in, queue_out)

    assert queue_out.put._mock_call_args_list[0].args == (0,)
    assert queue_out.put._mock_call_args_list[1].args == (None,)

    assert os.listdir(cache_dir) == ["a.txt"]


class DataModuleImage(LightningDataModule):
    def prepare_dataset_structure(self, src_dir: str, filepaths: List[str]) -> List[Any]:
        assert len(filepaths) == 30
        return filepaths

    def prepare_item(self, item):
        return item


@pytest.mark.parametrize("delete_cached_files", [False, True])
@pytest.mark.parametrize("fast_dev_run", [False, True])
@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
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
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    dataset_optimizer = DatasetOptimizer(
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
    dataset_optimizer.run(DataModuleImage())

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
@pytest.mark.skipif(
    condition=not _PIL_AVAILABLE or sys.platform == "win32" or sys.platform == "darwin", reason="Requires: ['pil']"
)
def test_data_optimizer_distributed(fast_dev_run, delete_cached_files, tmpdir, monkeypatch):
    """This test ensures the data optimizer works in a fully distributed settings."""

    from PIL import Image

    imgs = []
    for i in range(30):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(tmpdir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)

    remote_dst_dir = os.path.join(tmpdir, "dst")
    os.makedirs(remote_dst_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_1")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    dataset_optimizer = DatasetOptimizer(
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
    dataset_optimizer.run(DataModuleImage())

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
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    dataset_optimizer = DatasetOptimizer(
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
    dataset_optimizer.run(DataModuleImage())

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


class DataModule(LightningDataModule):
    @staticmethod
    def prepare_dataset_structure(src_dir: str, filepaths: List[str]) -> List[Any]:
        return [os.path.join(src_dir, "dummy2")]

    @staticmethod
    def prepare_item(filepath):
        for _ in range(100):
            yield torch.randint(0, 1000, (np.random.randint(0, 1000),)).to(torch.int)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_optimizer_nlp(tmpdir, monkeypatch):
    seed_everything(42)

    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(tmpdir))

    with open(os.path.join(tmpdir, "dummy.txt"), "w") as f:
        f.write("Hello World !")

    dataset_optimizer = DatasetOptimizer(
        name="dummy2", src_dir=tmpdir, num_workers=1, num_downloaders=1, chunk_size=1024 * 11
    )
    dataset_optimizer.run(DataModule())


def test_data_optimizer_api(tmpdir):
    dataset_optimizer = DatasetOptimizer(
        name="dummy2", src_dir=tmpdir, num_workers=1, num_downloaders=1, chunk_size=1024 * 11
    )
    with pytest.raises(ValueError, match="prepare_dataset_structure"):
        dataset_optimizer.run(None)
