import hashlib
import os
from multiprocessing import Process, Queue
from threading import Lock, Thread
from time import sleep
from typing import Any, Callable, List, Optional
from urllib import parse

import boto3
from torch.utils._pytree import tree_flatten, tree_unflatten
from tqdm import tqdm

from lightning.data.cache import Cache


def _download_data(queue_in: Queue, queue_out: Queue) -> None:
    s3 = boto3.client("s3")
    while True:
        r = queue_in.get()

        if r is None:
            queue_out.put(None)
            return

        index, remote_path, local_path = r

        obj = parse.urlparse(remote_path)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_path}")

        dirpath = os.path.dirname(local_path)

        os.makedirs(dirpath, exist_ok=True)

        with open(local_path, "wb") as f:
            s3.download_fileobj(obj.netloc, obj.path.lstrip("/"), f)

        queue_out.put(index)


def cleanup(queue_in: Queue):
    while True:
        r = queue_in.get()

        if r is None:
            return

        os.remove(r)


class DataThread(Thread):
    def __init__(
        self,
        index: int,
        prepare_item: Callable,
        root: str,
        remote_root: str,
        items: List[Any],
        num_downloaders: int,
        cleanup: bool,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
        compression: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self.index = index
        self.prepare_item = prepare_item
        self.root = root
        if root.startswith("/"):
            root = root[1:]
        self.remote_root = remote_root
        self.cache_dir = os.path.join("/cache", root)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.items = items
        self.cleanup = cleanup
        self.lock = Lock()
        self.paths = []
        self.downloaders = []
        self.to_download_queues = []
        self.download_is_ready_queue = Queue()
        self.remove_queue = Queue()
        self.counter = 0
        self.num_downloaders = num_downloaders

        algo = hashlib.new("sha256")
        algo.update(root.encode("utf-8"))
        _hash = algo.hexdigest()

        cache_dir = f"/cache/{_hash}_{index}"
        os.makedirs(cache_dir, exist_ok=True)

        self.cache = Cache(cache_dir, chunk_bytes=chunk_bytes, chunk_size=chunk_size, compression=compression)

    def __len__(self):
        with self.lock:
            return self.counter

    def run(self):
        self.collect_paths()

        for _ in range(self.num_downloaders):
            to_download_queue = Queue()
            p = Process(target=_download_data, args=(to_download_queue, self.download_is_ready_queue))
            p.start()
            self.downloaders.append(p)
            self.to_download_queues.append(to_download_queue)

        if self.cleanup:
            self.cleaner = Process(target=cleanup, args=(self.remove_queue))
            self.cleaner.start()

        for index, path in enumerate(self.paths):
            self.to_download_queues[index % self.num_downloaders].put((index, *path))

        for downloader_index in range(self.num_downloaders):
            self.to_download_queues[downloader_index].put(None)

        is_none = 0
        while True:
            r = self.download_is_ready_queue.get()
            if r is None:
                is_none += 1
                if is_none == self.num_downloaders:
                    self.remove_queue.put(None)
                    self.cache.done()
                    return
                continue

            # TODO: Add support non-ordered
            self.cache[r] = self.prepare_item(self.items[r])
            with self.lock:
                self.counter += 1
            if self.cleanup:
                self.remove_queue.put(r)

    def collect_paths(self):
        items = []
        for item in self.items:
            flattened_item, spec = tree_flatten(item)

            indexed_paths = {
                index: element
                for index, element in enumerate(flattened_item)
                if isinstance(element, str) and element.startswith(self.root)
            }

            if len(indexed_paths) == 0:
                raise ValueError(f"The provided item {item} didn't contain any filepaths.")

            for index, path in indexed_paths.items():
                tmp_path = path.replace(self.root, self.cache_dir)
                remote_path = path.replace(self.root, self.remote_root)
                self.paths.append((remote_path, tmp_path))
                flattened_item[index] = tmp_path

            items.append(tree_unflatten(flattened_item, spec))

        self.items = items


class DataOptimizer:
    def __init__(
        self,
        setup: Callable,
        prepare_item: Optional[Callable] = None,
        num_workers: int = os.cpu_count(),
        num_downloaders: int = 2,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
        compression: Optional[str] = None,
        cleanup: bool = False,
    ):
        self.setup = setup
        self.prepare_item = prepare_item
        self.num_workers = num_workers
        self.num_downloaders = num_downloaders
        self.chunk_size = chunk_size
        self.chunk_bytes = chunk_bytes
        self.cleanup = cleanup
        self.compression = compression
        self.workers = []

    def run(self, root: str, remote_root: str) -> None:
        filepaths = self._cache_list_filepaths(root)
        user_items = self.setup(root, filepaths)

        worker_size = len(user_items) // self.num_workers
        workers_user_items = []
        for worker_idx in range(self.num_workers):
            is_last = worker_idx == self.num_workers - 1
            start = worker_idx * worker_size
            end = len(user_items) if is_last else (worker_idx + 1) * worker_size
            workers_user_items.append(user_items[start:end])

        for worker_idx, worker_user_items in enumerate(workers_user_items):
            self.workers.append(
                DataThread(
                    worker_idx,
                    self.prepare_item,
                    root,
                    remote_root,
                    worker_user_items.tolist(),
                    self.num_downloaders,
                    self.cleanup,
                    self.chunk_size,
                    self.chunk_bytes,
                    self.compression,
                )
            )
            self.workers[-1].start()

        num_items = len(user_items)
        current_total = 0
        with tqdm(total=num_items) as pbar:
            while True:
                new_total = sum([len(w) for w in self.workers])
                pbar.update(new_total - current_total)
                current_total = new_total
                sleep(1)

    def _cache_list_filepaths(self, root: str) -> List[str]:
        algo = hashlib.new("sha256")
        algo.update(root.encode("utf-8"))
        _hash = algo.hexdigest()

        filepath = f"/cache/{_hash}.txt"

        if os.path.exists(filepath):
            lines = []
            with open(filepath) as f:
                for line in f.readlines():
                    lines.append(line.replace("\n", ""))
            return lines

        filepaths = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                filepaths.append(os.path.join(dirpath, filename))

        with open(filepath, "w") as f:
            for filepath in filepaths:
                f.write(f"{filepath}\n")

        return filepaths
