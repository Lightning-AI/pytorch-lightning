import os
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, List
from unittest import mock

import numpy as np
import pytest
import torch
from lightning import seed_everything
from lightning.data.processing import data_processor as data_processor_module
from lightning.data.processing import functions
from lightning.data.processing.data_processor import (
    DataChunkRecipe,
    DataProcessor,
    DataTransformRecipe,
    _download_data_target,
    _get_item_filesizes,
    _map_items_to_workers_sequentially,
    _map_items_to_workers_weighted,
    _remove_target,
    _upload_fn,
    _wait_for_disk_usage_higher_than_threshold,
    _wait_for_file_to_exist,
)
from lightning.data.processing.functions import LambdaDataTransformRecipe, map, optimize
from lightning.data.streaming import resolver
from lightning.data.streaming.cache import Cache, Dir
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_upload_fn(tmpdir):
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    remote_output_dir = os.path.join(tmpdir, "remote_output_dir")
    os.makedirs(remote_output_dir, exist_ok=True)

    filepath = os.path.join(input_dir, "a.txt")

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

    assert os.listdir(remote_output_dir) == []

    _upload_fn(upload_queue, remove_queue, cache_dir, Dir(path=remote_output_dir, url=remote_output_dir))

    assert os.listdir(remote_output_dir) == ["a.txt"]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_upload_s3_fn(tmpdir, monkeypatch):
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    remote_output_dir = os.path.join(tmpdir, "remote_output_dir")
    os.makedirs(remote_output_dir, exist_ok=True)

    filepath = os.path.join(input_dir, "a.txt")

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

    s3_client = mock.MagicMock()

    called = False

    def copy_file(local_filepath, *args):
        nonlocal called
        called = True
        from shutil import copyfile

        copyfile(local_filepath, os.path.join(remote_output_dir, os.path.basename(local_filepath)))

    s3_client.client.upload_file = copy_file

    monkeypatch.setattr(data_processor_module, "S3Client", mock.MagicMock(return_value=s3_client))

    assert os.listdir(remote_output_dir) == []

    assert not called

    _upload_fn(upload_queue, remove_queue, cache_dir, Dir(path=remote_output_dir, url="s3://url"))

    assert called

    assert len(paths) == 0

    assert os.listdir(remote_output_dir) == ["a.txt"]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_remove_target(tmpdir):
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    filepath = os.path.join(cache_dir, "a.txt")

    with open(filepath, "w") as f:
        f.write("HERE")

    filepath = os.path.join(input_dir, "a.txt")

    queue_in = mock.MagicMock()

    paths = [filepath, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return [value]

    queue_in.get = fn

    assert os.listdir(cache_dir) == ["a.txt"]

    _remove_target(Dir(path=input_dir), cache_dir, queue_in)

    assert os.listdir(cache_dir) == []


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
@mock.patch("lightning.data.processing.data_processor._wait_for_disk_usage_higher_than_threshold")
def test_download_data_target(wait_for_disk_usage_higher_than_threshold_mock, tmpdir):
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    remote_input_dir = os.path.join(tmpdir, "remote_input_dir")
    os.makedirs(remote_input_dir, exist_ok=True)

    with open(os.path.join(remote_input_dir, "a.txt"), "w") as f:
        f.write("HERE")

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    queue_in = mock.MagicMock()

    paths = [os.path.join(input_dir, "a.txt"), None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return (0, [value])

    queue_in.get = fn

    queue_out = mock.MagicMock()
    _download_data_target(Dir(input_dir, remote_input_dir), cache_dir, queue_in, queue_out)

    assert queue_out.put._mock_call_args_list[0].args == (0,)
    assert queue_out.put._mock_call_args_list[1].args == (None,)

    assert os.listdir(cache_dir) == ["a.txt"]

    wait_for_disk_usage_higher_than_threshold_mock.assert_called()


def test_wait_for_disk_usage_higher_than_threshold():
    disk_usage_mock = mock.Mock(side_effect=[mock.Mock(free=10e9), mock.Mock(free=10e9), mock.Mock(free=10e11)])
    with mock.patch("lightning.data.processing.data_processor.shutil.disk_usage", disk_usage_mock):
        _wait_for_disk_usage_higher_than_threshold("/", 10, sleep_time=0)
    assert disk_usage_mock.call_count == 3


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_wait_for_file_to_exist():
    import botocore

    s3 = mock.MagicMock()
    obj = mock.MagicMock()
    raise_error = [True, True, False]

    def fn(*_, **__):
        value = raise_error.pop(0)
        if value:
            raise botocore.exceptions.ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return

    s3.client.head_object = fn

    _wait_for_file_to_exist(s3, obj, sleep_time=0.01)

    assert len(raise_error) == 0

    def fn(*_, **__):
        raise ValueError("HERE")

    s3.client.head_object = fn

    with pytest.raises(ValueError, match="HERE"):
        _wait_for_file_to_exist(s3, obj, sleep_time=0.01)


def test_cache_dir_cleanup(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "chunks")
    cache_data_dir = os.path.join(tmpdir, "data")

    os.makedirs(cache_dir)

    with open(os.path.join(cache_dir, "a.txt"), "w") as f:
        f.write("Hello World !")

    assert os.listdir(cache_dir) == ["a.txt"]

    data_processor = DataProcessor(input_dir=str(tmpdir))
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(cache_dir))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", str(cache_data_dir))
    data_processor._cleanup_cache()

    assert os.listdir(cache_dir) == []


def test_map_items_to_workers_weighted(monkeypatch):
    seed_everything(42)

    workers_user_items = _map_items_to_workers_weighted(1, list(range(5)))
    assert workers_user_items == [[1, 4, 2, 0, 3]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(5)))
    assert workers_user_items == [[2, 4, 0], [3, 1]]
    workers_user_items = _map_items_to_workers_weighted(3, list(range(5)))
    assert workers_user_items == [[0, 3], [4, 1], [2]]
    workers_user_items = _map_items_to_workers_weighted(4, list(range(5)))
    assert workers_user_items == [[4, 0], [1], [2], [3]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_weighted(1, list(range(5)))
    assert workers_user_items == [[2, 0, 4]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(5)))
    assert workers_user_items == [[0, 4], [1]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    workers_user_items = _map_items_to_workers_weighted(1, list(range(5)))
    assert workers_user_items == [[3, 1]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(5)))
    assert workers_user_items == [[2], [3]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "4")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_weighted(1, list(range(32)))
    assert workers_user_items == [[0, 24, 28, 4, 16, 20, 8, 12]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(32)))
    assert workers_user_items == [[24, 16, 0, 8], [1, 17, 9, 25]]
    workers_user_items = _map_items_to_workers_weighted(3, list(range(32)))
    assert workers_user_items == [[24, 12, 0], [13, 25, 1], [14, 2, 26]]
    workers_user_items = _map_items_to_workers_weighted(4, list(range(32)))
    assert workers_user_items == [[16, 0], [1, 17], [2, 18], [3, 19]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "4")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "3")
    workers_user_items = _map_items_to_workers_weighted(1, list(range(32)))
    assert workers_user_items == [[3, 7, 19, 31, 11, 23, 27, 15]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(32)))
    assert workers_user_items == [[14, 22, 6, 30], [15, 31, 23, 7]]
    workers_user_items = _map_items_to_workers_weighted(3, list(range(32)))
    assert workers_user_items == [[21, 9], [22, 10], [23, 11]]
    workers_user_items = _map_items_to_workers_weighted(4, list(range(32)))
    assert workers_user_items == [[12, 28], [13, 29], [30, 14], [15, 31]]


def test_map_items_to_workers_sequentially(monkeypatch):
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)))
    assert workers_user_items == [list(range(5))]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)))
    assert workers_user_items == [[0, 1], [2, 3, 4]]
    workers_user_items = _map_items_to_workers_sequentially(3, list(range(5)))
    assert workers_user_items == [[0], [1, 2], [3, 4]]
    workers_user_items = _map_items_to_workers_sequentially(4, list(range(5)))
    assert workers_user_items == [[0], [1], [2], [3, 4]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)))
    assert workers_user_items == [[0, 1]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)))
    assert workers_user_items == [[0], [1]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)))
    assert workers_user_items == [[2, 3, 4]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)))
    assert workers_user_items == [[2], [3, 4]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "4")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(32)))
    assert workers_user_items == [[0, 1, 2, 3, 4, 5, 6, 7]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(32)))
    assert workers_user_items == [[0, 1, 2, 3], [4, 5, 6, 7]]
    workers_user_items = _map_items_to_workers_sequentially(3, list(range(32)))
    assert workers_user_items == [[0, 1], [2, 3], [4, 5]]
    workers_user_items = _map_items_to_workers_sequentially(4, list(range(32)))
    assert workers_user_items == [[0, 1], [2, 3], [4, 5], [6, 7]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "4")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "3")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(32)))
    assert workers_user_items == [[24, 25, 26, 27, 28, 29, 30, 31]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(32)))
    assert workers_user_items == [[24, 25, 26, 27], [28, 29, 30, 31]]
    workers_user_items = _map_items_to_workers_sequentially(3, list(range(32)))
    assert workers_user_items == [[23, 24, 25], [26, 27, 28], [29, 30, 31]]
    workers_user_items = _map_items_to_workers_sequentially(4, list(range(32)))
    assert workers_user_items == [[24, 25], [26, 27], [28, 29], [30, 31]]


class CustomDataChunkRecipe(DataChunkRecipe):
    def prepare_structure(self, input_dir: str) -> List[Any]:
        filepaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
        assert len(filepaths) == 30
        return filepaths

    def prepare_item(self, item):
        return item


@pytest.mark.parametrize("delete_cached_files", [True])
@pytest.mark.parametrize("fast_dev_run", [10])
@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processsor(fast_dev_run, delete_cached_files, tmpdir, monkeypatch):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir)

    imgs = []
    for i in range(30):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    cache_data_dir = os.path.join(tmpdir, "cache", "data")
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_data_dir)

    data_processor = DataProcessor(
        input_dir=input_dir,
        num_workers=2,
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
    )
    data_processor.run(CustomDataChunkRecipe(chunk_size=2))

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

    chunks = fast_dev_run_enabled_chunks if fast_dev_run == 10 else fast_dev_run_disabled_chunks

    assert sorted(os.listdir(cache_dir)) == chunks

    files = []
    for _, _, filenames in os.walk(os.path.join(cache_dir, "data")):
        files.extend(filenames)

    expected = (0 if delete_cached_files else 20) if fast_dev_run == 10 else (0 if delete_cached_files else 30)
    assert len(files) == expected


class TestDataProcessor(DataProcessor):
    def _broadcast_object(self, obj: Any) -> Any:
        return obj


@pytest.mark.parametrize("delete_cached_files", [False])
@pytest.mark.parametrize("fast_dev_run", [False])
@pytest.mark.skipif(
    condition=(not _PIL_AVAILABLE or sys.platform == "win32" or sys.platform == "linux"), reason="Requires: ['pil']"
)
def test_data_processsor_distributed(fast_dev_run, delete_cached_files, tmpdir, monkeypatch):
    """This test ensures the data optimizer works in a fully distributed settings."""

    seed_everything(42)

    monkeypatch.setattr(data_processor_module.os, "_exit", mock.MagicMock())

    _create_dataset_mock = mock.MagicMock()

    monkeypatch.setattr(data_processor_module, "_create_dataset", _create_dataset_mock)

    from PIL import Image

    input_dir = os.path.join(tmpdir, "dataset")
    os.makedirs(input_dir)

    imgs = []
    for i in range(30):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)

    remote_output_dir = os.path.join(tmpdir, "dst")
    os.makedirs(remote_output_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_1")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    data_cache_dir = os.path.join(tmpdir, "data_cache_1")
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    data_processor = TestDataProcessor(
        input_dir=input_dir,
        num_workers=2,
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
        output_dir=remote_output_dir,
        num_uploaders=1,
        num_downloaders=1,
    )
    data_processor.run(CustomDataChunkRecipe(chunk_size=2))

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

    assert sorted(os.listdir(remote_output_dir)) == fast_dev_run_disabled_chunks_0

    cache_dir = os.path.join(tmpdir, "cache_2")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    data_processor = TestDataProcessor(
        input_dir=input_dir,
        num_workers=2,
        num_uploaders=1,
        num_downloaders=1,
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
        output_dir=remote_output_dir,
    )
    data_processor.run(CustomDataChunkRecipe(chunk_size=2))

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

    expected = sorted(fast_dev_run_disabled_chunks_0 + fast_dev_run_disabled_chunks_1 + ["1-index.json"])

    assert sorted(os.listdir(remote_output_dir)) == expected

    _create_dataset_mock.assert_not_called()


class TextTokenizeRecipe(DataChunkRecipe):
    def prepare_structure(self, input_dir: str) -> List[Any]:
        return [os.path.join(input_dir, "dummy.txt")]

    def prepare_item(self, filepath):
        for _ in range(100):
            yield torch.randint(0, 1000, (np.random.randint(0, 1000),)).to(torch.int)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_processsor_nlp(tmpdir, monkeypatch):
    seed_everything(42)

    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", os.path.join(tmpdir, "chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", os.path.join(tmpdir, "data"))

    with open(os.path.join(tmpdir, "dummy.txt"), "w") as f:
        f.write("Hello World !")

    data_processor = DataProcessor(input_dir=str(tmpdir), num_workers=1, num_downloaders=1)
    data_processor.run(TextTokenizeRecipe(chunk_size=1024 * 11))

    data_processor_more_wokers = DataProcessor(input_dir=str(tmpdir), num_workers=2, num_downloaders=1)
    data_processor_more_wokers.run(TextTokenizeRecipe(chunk_size=1024 * 11))


class ImageResizeRecipe(DataTransformRecipe):
    def prepare_structure(self, input_dir: str):
        filepaths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
        return [filepath for filepath in filepaths if os.path.isfile(filepath)]

    def prepare_item(self, filepath: Any, output_dir: str, is_last) -> None:
        from PIL import Image

        img = Image.open(filepath)
        img = img.resize((12, 12))
        assert os.path.exists(output_dir)
        img.save(os.path.join(output_dir, os.path.basename(filepath)))


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_process_transform(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir)

    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "output_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    data_processor = DataProcessor(
        input_dir=input_dir,
        num_workers=1,
        output_dir=output_dir,
        fast_dev_run=False,
    )
    data_processor.run(ImageResizeRecipe())

    assert sorted(os.listdir(output_dir)) == ["0.JPEG", "1.JPEG", "2.JPEG", "3.JPEG", "4.JPEG"]

    from PIL import Image

    img = Image.open(os.path.join(output_dir, "0.JPEG"))
    assert img.size == (12, 12)


def map_fn(filepath, output_dir):
    from PIL import Image

    img = Image.open(filepath)
    img = img.resize((12, 12))
    assert os.path.exists(output_dir)
    img.save(os.path.join(output_dir, os.path.basename(filepath)))


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_map(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    inputs = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    inputs = [filepath for filepath in inputs if os.path.isfile(filepath)]

    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    map(map_fn, inputs, output_dir=output_dir, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["0.JPEG", "1.JPEG", "2.JPEG", "3.JPEG", "4.JPEG"]

    from PIL import Image

    img = Image.open(os.path.join(output_dir, "0.JPEG"))
    assert img.size == (12, 12)


def optimize_fn(filepath):
    from PIL import Image

    return [Image.open(filepath), os.path.basename(filepath)]


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_optimize(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "output_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    inputs = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    inputs = [filepath for filepath in inputs if os.path.isfile(filepath)]

    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    optimize(optimize_fn, inputs, output_dir=output_dir, chunk_size=2, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["chunk-0-0.bin", "chunk-0-1.bin", "chunk-0-2.bin", "index.json"]

    cache = Cache(output_dir, chunk_size=1)
    assert len(cache) == 5


def generate_data(index, shift=None):
    yield from range(index + shift if shift else 0)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_optimize_yield(monkeypatch, tmpdir):
    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "output_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    optimize(partial(generate_data, shift=2), [0, 1], output_dir=output_dir, chunk_size=2, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["chunk-0-0.bin", "chunk-0-1.bin", "chunk-0-2.bin", "index.json"]


class Optimize:
    def __call__(self, filepath):
        from PIL import Image

        return [Image.open(filepath), os.path.basename(filepath)]


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_optimize_class(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    inputs = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    inputs = [filepath for filepath in inputs if os.path.isfile(filepath)]

    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    optimize(Optimize(), inputs, output_dir=output_dir, chunk_size=2, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["chunk-0-0.bin", "chunk-0-1.bin", "chunk-0-2.bin", "index.json"]

    cache = Cache(output_dir, chunk_size=1)
    assert len(cache) == 5


class OptimizeYield:
    def __call__(self, filepath):
        from PIL import Image

        for _ in range(1):
            yield [Image.open(filepath), os.path.basename(filepath)]


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_optimize_class_yield(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    inputs = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    inputs = [filepath for filepath in inputs if os.path.isfile(filepath)]

    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    optimize(OptimizeYield(), inputs, output_dir=output_dir, chunk_size=2, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["chunk-0-0.bin", "chunk-0-1.bin", "chunk-0-2.bin", "index.json"]

    cache = Cache(output_dir, chunk_size=1)
    assert len(cache) == 5


def test_lambda_transform_recipe(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.cuda.device_count.return_value = 3

    monkeypatch.setattr(functions, "torch", torch_mock)
    monkeypatch.setenv("DATA_OPTIMIZER_GLOBAL_RANK", 2)

    called = False

    def fn(output_dir, item, device):
        nonlocal called
        assert device == "cuda:2"
        called = True

    data_recipe = LambdaDataTransformRecipe(fn, range(1))

    data_recipe.prepare_item(1, "", False)
    assert called


def test_lambda_transform_recipe_class(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.cuda.device_count.return_value = 3

    monkeypatch.setattr(functions, "torch", torch_mock)
    monkeypatch.setenv("DATA_OPTIMIZER_GLOBAL_RANK", 2)

    called = False

    class Transform:
        def __call__(self, item, output_dir, device):
            nonlocal called
            assert device == "cuda:2"
            called = True

    data_recipe = LambdaDataTransformRecipe(Transform(), range(1))
    data_recipe.prepare_item(1, "", False)
    assert called


def _generate_file_with_size(file_path, num_bytes):
    assert num_bytes % 8 == 0
    content = bytearray(random.getrandbits(8) for _ in range(num_bytes))
    with open(file_path, "wb") as file:
        file.write(content)


def test_get_item_filesizes(tmp_path):
    _generate_file_with_size(tmp_path / "file1", 32)
    _generate_file_with_size(tmp_path / "file2", 64)
    _generate_file_with_size(tmp_path / "file3", 128)
    _generate_file_with_size(tmp_path / "file4", 256)

    items = [
        # not a path
        "not a path",
        # single file path
        str(tmp_path / "file1"),
        # tuple: one file path
        (1, 2, str(tmp_path / "file2")),
        # list: two file paths
        [str(tmp_path / "file2"), None, str(tmp_path / "file3")],
        # list: one file path exists, one does not
        [str(tmp_path / "other" / "other"), None, str(tmp_path / "file4")],
        # dict: with file path
        {"file": str(tmp_path / "file4"), "data": "not file"},
    ]
    num_bytes = _get_item_filesizes(items, base_path=str(tmp_path))
    assert num_bytes == [0, 32, 64, 64 + 128, 256, 256]

    with open(tmp_path / "empty_file", "w"):
        pass
    assert os.path.getsize(tmp_path / "empty_file") == 0
    with pytest.raises(RuntimeError, match="has 0 bytes!"):
        _get_item_filesizes([str(tmp_path / "empty_file")])


def map_fn_index(index, output_dir):
    with open(os.path.join(output_dir, f"{index}.JPEG"), "w") as f:
        f.write("Hello")


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_map_without_input_dir(monkeypatch, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    map(
        map_fn_index,
        list(range(5)),
        output_dir=output_dir,
        num_workers=1,
        reorder_files=True,
        weights=[1 for _ in range(5)],
    )

    assert sorted(os.listdir(output_dir)) == ["0.JPEG", "1.JPEG", "2.JPEG", "3.JPEG", "4.JPEG"]


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_map_weights_mismatch(monkeypatch, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    with pytest.raises(ValueError, match="The provided weights length"):
        map(map_fn_index, list(range(5)), output_dir=output_dir, num_workers=1, reorder_files=True, weights=[1])


def map_fn_index_folder(index, output_dir):
    os.makedirs(os.path.join(output_dir, str(index)))
    with open(os.path.join(output_dir, str(index), f"{index}.JPEG"), "w") as f:
        f.write("Hello")


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_map_without_input_dir_and_folder(monkeypatch, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    map(map_fn_index_folder, list(range(5)), output_dir=output_dir, num_workers=1, reorder_files=True)

    assert sorted(os.listdir(output_dir)) == ["0", "1", "2", "3", "4"]
    assert os.path.exists(os.path.join(output_dir, "0", "0.JPEG"))


def map_fn_map_non_absolute(path, output_dir):
    absolute_path = str(Path(path).absolute())
    assert absolute_path == path, (absolute_path, path)

    with open(os.path.join(output_dir, os.path.basename(path)), "w") as f:
        f.write("Hello World")


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows")
def test_data_processing_map_non_absolute_path(monkeypatch, tmpdir):
    monkeypatch.chdir(str(tmpdir))

    for i in range(5):
        with open(f"./{i}.txt", "w") as f:
            f.write("Hello World")

    assert sorted(os.listdir(tmpdir)) == ["0.txt", "1.txt", "2.txt", "3.txt", "4.txt"]

    map(
        map_fn_map_non_absolute,
        [f"{i}.txt" for i in range(5)],
        output_dir="./output_dir",
        num_workers=1,
        reorder_files=True,
    )

    assert sorted(os.listdir(tmpdir)) == ["0.txt", "1.txt", "2.txt", "3.txt", "4.txt", "output_dir"]
    assert sorted(os.listdir(os.path.join(tmpdir, "output_dir"))) == ["0.txt", "1.txt", "2.txt", "3.txt", "4.txt"]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_map_error_when_not_empty(monkeypatch):
    boto3 = mock.MagicMock()
    client_s3_mock = mock.MagicMock()
    client_s3_mock.list_objects_v2.return_value = {"KeyCount": 1, "Contents": []}
    boto3.client.return_value = client_s3_mock
    monkeypatch.setattr(resolver, "boto3", boto3)

    with pytest.raises(RuntimeError, match="data and datasets are meant to be immutable"):
        map(
            map_fn,
            [0, 1],
            output_dir=Dir(path=None, url="s3://bucket"),
            error_when_not_empty=True,
        )

    monkeypatch.setattr(data_processor_module, "_IS_IN_STUDIO", True)

    with pytest.raises(OSError, match="cache"):
        map(
            map_fn,
            [0, 1],
            output_dir=Dir(path=None, url="s3://bucket"),
            error_when_not_empty=False,
        )

def map_fn_is_last(index, output_dir, is_last):
    with open(os.path.join(output_dir, f"{index}_{is_last}.txt"), "w") as f:
        f.write("here")


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
@pytest.mark.parametrize(
    ("num_workers", "expected"),
    [
        (1, ['0_False.txt', '1_False.txt', '2_False.txt', '3_False.txt', '4_True.txt']),
        (2, ['0_False.txt', '1_True.txt', '2_False.txt', '3_False.txt', '4_True.txt']),
    ],
)
def test_map_is_last(num_workers, expected, tmpdir):
    map(
        map_fn_is_last,
        list(range(5)),
        output_dir=str(tmpdir),
        error_when_not_empty=False,
        num_workers=num_workers,
    )

    assert sorted(os.listdir(tmpdir)) == expected
