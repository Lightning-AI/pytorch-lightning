import os
import shutil

from lightning.data.streaming.cache import Cache
from lightning.data.streaming.config import ChunkedIndex
from lightning_cloud.resolver import Dir


def test_reader_chunk_removal(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    remote_dir = os.path.join(tmpdir, "remote_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=28020)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    shutil.copytree(cache_dir, remote_dir)
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    for i in range(25):
        index = ChunkedIndex(i, cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert len(os.listdir(cache_dir)) == 14

    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=2800)

    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    generated = []
    for i in range(25):
        generated.append([i, len(os.listdir(cache_dir))])
        index = ChunkedIndex(i, cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert generated == [
        [0, 0],
        [1, 2],
        [2, 2],
        [3, 2],
        [4, 2],
        [5, 2],
        [6, 2],
        [7, 2],
        [8, 2],
        [9, 2],
        [10, 2],
        [11, 2],
        [12, 2],
        [13, 2],
        [14, 2],
        [15, 2],
        [16, 2],
        [17, 2],
        [18, 2],
        [19, 2],
        [20, 2],
        [21, 2],
        [22, 2],
        [23, 2],
        [24, 2],
    ]

    assert len(os.listdir(cache_dir)) == 2
