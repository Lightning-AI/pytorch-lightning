import os
import shutil

import numpy as np
from lightning.data.streaming.cache import Cache
from lightning.data.streaming.config import ChunkedIndex
from lightning.data.streaming.item_loader import PyTreeLoader
from lightning.data.streaming.reader import PrepareChunksThread, _get_folder_size
from lightning_cloud.resolver import Dir


def test_reader_chunk_removal(tmpdir):
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
        [3, 3],
        [4, 3],
        [5, 3],
        [6, 3],
        [7, 3],
        [8, 3],
        [9, 3],
        [10, 3],
        [11, 3],
        [12, 3],
        [13, 3],
        [14, 3],
        [15, 3],
        [16, 3],
        [17, 3],
        [18, 3],
        [19, 3],
        [20, 3],
        [21, 3],
        [22, 3],
        [23, 3],
        [24, 3],
    ]

    assert len(os.listdir(cache_dir)) == 3


def test_get_folder_size(tmpdir):
    array = np.zeros((10, 10))

    np.save(os.path.join(tmpdir, "array_1.npy"), array)
    np.save(os.path.join(tmpdir, "array_2.npy"), array)

    assert _get_folder_size(tmpdir) == 928 * 2


def test_prepare_chunks_thread(tmpdir):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=2, max_cache_size=28020)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    cache._reader._try_load_config()

    thread = PrepareChunksThread(cache._reader.config, item_loader=PyTreeLoader(), max_cache_size=1)
    assert thread._delete_chunks_when_processed

    thread = PrepareChunksThread(cache._reader.config, item_loader=PyTreeLoader(), max_cache_size=10000)
    assert not thread._delete_chunks_when_processed
