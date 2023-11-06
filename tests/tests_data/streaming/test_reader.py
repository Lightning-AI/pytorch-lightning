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

    for i in range(25):
        assert cache[i] == i

    assert len(os.listdir(cache_dir)) == 3
