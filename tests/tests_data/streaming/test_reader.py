import os
import shutil
from unittest import mock

from lightning.data.streaming import reader
from lightning.data.streaming.cache import Cache
from lightning.data.utilities import disk
from lightning_cloud.resolver import Dir


def test_reader_chunk_removal(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    remote_dir = os.path.join(tmpdir, "remote_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    monkeypatch.setattr(reader, "_get_available_amount_of_bytes", lambda: 53687091201)

    shutil.copytree(cache_dir, remote_dir)
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    for i in range(25):
        assert cache[i] == i

    assert len(os.listdir(cache_dir)) == 14

    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    monkeypatch.setattr(reader, "_get_available_amount_of_bytes", lambda: 0)

    for i in range(25):
        assert cache[i] == i

    assert len(os.listdir(cache_dir)) == 3


def test_disk_get_available_amount_of_bytes(monkeypatch):
    result = b"Filesystem     1024-blocks      Used Available Capacity iused      ifree %iused  Mounted on\n/dev/disk3s1s1   971350180   9662852 397071612     3%  390143 3970716120    0%   /\n"  # noqa: E501
    subprocess_mock = mock.MagicMock()
    subprocess_mock.check_output.return_value = result
    monkeypatch.setattr(disk, "subprocess", subprocess_mock)
    assert disk._get_available_amount_of_bytes() == 397071612
    subprocess_mock.check_output.assert_called()
