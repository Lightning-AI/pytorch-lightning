import os
import shutil

from lightning.data.streaming import reader
from lightning.data.streaming.cache import Cache
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
