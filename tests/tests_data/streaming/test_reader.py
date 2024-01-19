import os
import shutil
from time import sleep

import numpy as np
from lightning.data.streaming import reader
from lightning.data.streaming.cache import Cache
from lightning.data.streaming.config import ChunkedIndex
from lightning.data.streaming.item_loader import PyTreeLoader
from lightning.data.streaming.reader import _END_TOKEN, PrepareChunksThread, _get_folder_size
from lightning.data.streaming.resolver import Dir
from lightning.data.utilities.env import _DistributedEnv


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

    for i in range(25):
        assert len(os.listdir(cache_dir)) <= 3
        index = ChunkedIndex(i, cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert len(os.listdir(cache_dir)) == 3


def test_get_folder_size(tmpdir):
    array = np.zeros((10, 10))

    np.save(os.path.join(tmpdir, "array_1.npy"), array)
    np.save(os.path.join(tmpdir, "array_2.npy"), array)

    assert _get_folder_size(tmpdir) == 928 * 2


def test_prepare_chunks_thread_eviction(tmpdir, monkeypatch):
    monkeypatch.setattr(reader, "_LONG_DEFAULT_TIMEOUT", 0.1)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=2, max_cache_size=28020)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    cache._reader._try_load_config()

    assert len(os.listdir(cache_dir)) == 14

    thread = PrepareChunksThread(
        cache._reader.config, item_loader=PyTreeLoader(), distributed_env=_DistributedEnv(1, 1, 1), max_cache_size=10000
    )
    assert not thread._delete_chunks_when_processed

    thread = PrepareChunksThread(
        cache._reader.config, item_loader=PyTreeLoader(), distributed_env=_DistributedEnv(1, 1, 1), max_cache_size=1
    )
    assert thread._delete_chunks_when_processed

    thread.start()

    assert thread._pre_download_counter == 0

    thread.download([0, 1, 2, 3, 4, 5, _END_TOKEN])

    while thread._pre_download_counter == 0:
        sleep(0.01)

    assert not thread._has_exited

    for i in range(5):
        thread.delete([i])
        while len(os.listdir(cache_dir)) != 14 - (i + 1):
            sleep(0.01)

    assert thread._pre_download_counter <= 2

    assert len(os.listdir(cache_dir)) == 9
    assert thread._has_exited
    thread.join()
