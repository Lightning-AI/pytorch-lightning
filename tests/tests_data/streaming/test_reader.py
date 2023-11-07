import os
import shutil
from unittest import mock

from lightning.data.streaming import reader
from lightning.data.streaming.cache import Cache
from lightning_cloud.resolver import Dir


def test_reader_chunk_removal(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    remote_dir = os.path.join(tmpdir, "remote_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=53687091200)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    shutil_mock = mock.MagicMock()
    disk_usage = mock.MagicMock()
    disk_usage.total = 1230
    shutil_mock.disk_usage.return_value = disk_usage
    monkeypatch.setattr(reader, "shutil", shutil_mock)

    shutil.copytree(cache_dir, remote_dir)
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    for i in range(25):
        assert cache[i] == i

    assert len(os.listdir(cache_dir)) == 14

    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    shutil_mock = mock.MagicMock()
    disk_usage = mock.MagicMock()
    disk_usage.total = 536870912000
    shutil_mock.disk_usage.return_value = disk_usage
    monkeypatch.setattr(reader, "shutil", shutil_mock)

    expected = []
    for i in range(25):
        expected.append([i, len(os.listdir(cache_dir))])
        assert cache[i] == i

    assert expected == [
        [0, 0],
        [1, 1],
        [2, 1],
        [3, 2],
        [4, 2],
        [5, 3],
        [6, 3],
        [7, 4],
        [8, 4],
        [9, 5],
        [10, 5],
        [11, 6],
        [12, 6],
        [13, 7],
        [14, 7],
        [15, 8],
        [16, 8],
        [17, 9],
        [18, 9],
        [19, 10],
        [20, 10],
        [21, 11],
        [22, 11],  # Cleanup is triggered
        [23, 2],
        [24, 2],
    ]

    assert len(os.listdir(cache_dir)) == 3
