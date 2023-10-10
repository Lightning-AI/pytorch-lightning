import hashlib
import logging
import os
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Lock, Thread
from time import sleep, time
from typing import Any, Callable, List, Optional
from urllib import parse

import boto3
from torch.utils._pytree import tree_flatten, tree_unflatten
from tqdm import tqdm

from lightning.data.cache import Cache

logger = logging.Logger(__name__)


def _download_data(root: str, remote_root: str, cache_dir: str, queue_in: Queue, queue_out: Queue) -> None:
    s3 = boto3.client("s3")
    while True:
        r = queue_in.get()

        if r is None:
            queue_out.put(None)
            return

        index, paths = r

        # Check whether all the files are already downloaded
        if all(os.path.exists(p) for p in paths):
            queue_out.put(index)
            continue

        # Download all the required paths to unblock the current index
        for path in paths:
            remote_path = path.replace(root, remote_root)
            obj = parse.urlparse(remote_path)

            if obj.scheme != "s3":
                raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_path}")

            local_path = path.replace(root, cache_dir)

            dirpath = os.path.dirname(local_path)

            os.makedirs(dirpath, exist_ok=True)

            with open(local_path, "wb") as f:
                s3.download_fileobj(obj.netloc, obj.path.lstrip("/"), f)

        queue_out.put(index)


def remove(queue_in: Queue):
    while True:
        r = queue_in.get()

        if r is None:
            return

        os.remove(r)


class DataWorker(Thread):
    def __init__(
        self,
        index: int,
        start_index: int,
        prepare_item: Callable,
        root: str,
        remote_root: str,
        items: List[Any],
        num_downloaders: int,
        remove: bool,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
        compression: Optional[str] = None,
    ):
        """The DataWorker is responsible to process the user data."""
        super().__init__(daemon=True)
        self.index = index
        self.start_index = start_index
        self.prepare_item = prepare_item
        self.root = root
        self.remote_root = remote_root
        self.items = items
        self.num_downloaders = num_downloaders
        self.remove = remove
        self.chunk_bytes = chunk_bytes
        self.chunk_size = chunk_size
        self.compression = compression
        self._lock = Lock()
        self._paths = []
        self._downloaders = []
        self._to_download_queues = []
        self._download_is_ready_queue = Queue()
        self._remove_queue = Queue()
        self._collected_items = 0
        self._counter = 0

    def _create_cache(self):
        algo = hashlib.new("sha256")
        algo.update(self.root.encode("utf-8"))
        root_hash = algo.hexdigest()

        cache_dir = f"/cache/{root_hash}/w_{self.index}"
        os.makedirs(cache_dir, exist_ok=True)

        self.cache = Cache(
            cache_dir, chunk_bytes=self.chunk_bytes, chunk_size=self.chunk_size, compression=self.compression
        )

        self.cache_dir = f"/cache/{root_hash}/data"
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        with self._lock:
            return self._counter

    @property
    def collected_items(self):
        with self._lock:
            return self._collected_items

    def run(self):
        self._collect_paths()
        self._start_downloaders()
        self._start_remover()

        is_none = 0
        while True:
            r = self._download_is_ready_queue.get()
            if r is None:
                is_none += 1
                if is_none == self.num_downloaders:
                    self.remove_queue.put(None)
                    self.cache.done()
                    return
                continue

            self.cache[r + self.start_index] = self.prepare_item(self.items[r]) if self.prepare_item else self.items[r]

            with self._lock:
                self._counter += 1

            if self.remove:
                self._remove_queue.put(r)

    def _collect_paths(self):
        items = []
        for item in self.items:
            flattened_item, spec = tree_flatten(item)

            # For speed reasons, we assume starting with `self.root` is enough to be a real file.
            # Other alternative would be too slow.
            # TODO: Try using dictionary for higher accurary.
            indexed_paths = {
                index: element
                for index, element in enumerate(flattened_item)
                if isinstance(element, str) and element.startswith(self.root)  # For speed reasons
            }

            if len(indexed_paths) == 0:
                raise ValueError(f"The provided item {item} didn't contain any filepaths.")

            paths = []
            for index, path in indexed_paths.items():
                tmp_path = path.replace(self.root, self.cache_dir)
                flattened_item[index] = tmp_path
                paths.append(path)

            self._paths.append(path)

            items.append(tree_unflatten(flattened_item, spec))

            with self._lock:
                self._collected_items += 1

        self.items = items

    def _start_downloaders(self):
        for _ in range(self.num_downloaders):
            to_download_queue = Queue()
            p = Process(
                target=_download_data,
                args=(self.root, self.remote_root, self.cache_dir, to_download_queue, self._download_is_ready_queue),
            )
            p.start()
            self._downloaders.append(p)
            self._to_download_queues.append(to_download_queue)

        for index, path in enumerate(self._paths):
            self._to_download_queues[index % self.num_downloaders].put((index, *path))

        for downloader_index in range(self.num_downloaders):
            self._to_download_queues[downloader_index].put(None)

    def _start_remover(self):
        if self.remove:
            self.remover = Process(target=remove, args=(self._remove_queue))
            self.remover.start()


class DataProcessor:
    def __init__(
        self,
        setup: Callable,
        prepare_item: Optional[Callable] = None,
        num_workers: int = os.cpu_count() * 3,
        num_downloaders: int = 3,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = 1 << 26,
        compression: Optional[str] = None,
        remove: bool = False,
    ):
        self.setup = setup
        self.prepare_item = prepare_item
        self.num_workers = num_workers
        self.num_downloaders = num_downloaders
        self.chunk_size = chunk_size
        self.chunk_bytes = chunk_bytes
        self.remove = remove
        self.compression = compression
        self.workers = []

    def run(self, root: str, remote_root: str) -> None:
        t0 = time()
        logger.info("Setup started")

        # Get the filepaths
        root = str(Path(root).resolve())
        filepaths = self._cached_list_filepaths(root)
        num_filepaths = len(filepaths)

        # Call the setup method of the user
        user_items = self.setup(root, filepaths)

        # Associate the items to the workers
        worker_size = num_filepaths // self.num_workers
        workers_user_items = []
        begins = []
        for worker_idx in range(self.num_workers):
            is_last = worker_idx == self.num_workers - 1
            begin = worker_idx * worker_size
            end = num_filepaths if is_last else (worker_idx + 1) * worker_size
            workers_user_items.append(user_items[begin:end])
            begins.append(begin)

        logger.info(f"Setup finished in {round(time() - t0, 3)} seconds. Found {num_filepaths} items to process.")

        logger.info(f"Starting {self.num_workers} workers")

        num_items = len(user_items)
        current_total = 0
        with tqdm(total=num_items, smoothing=0) as pbar:
            for worker_idx, worker_user_items in enumerate(workers_user_items):
                new_total = sum([w.collected_items for w in self.workers])
                pbar.update(new_total - current_total)
                current_total = new_total
                thread = DataWorker(
                    worker_idx,
                    begins[worker_idx],
                    self.prepare_item,
                    root,
                    remote_root,
                    worker_user_items.tolist(),
                    self.num_downloaders,
                    self.remove,
                    self.chunk_size,
                    self.chunk_bytes,
                    self.compression,
                )
                thread.start()
                self.workers.append(thread)

            while True:
                new_total = sum([w.collected_items for w in self.workers])
                pbar.update(new_total - current_total)
                current_total = new_total
                sleep(1)
                if current_total == num_filepaths:
                    break

        logger.info("Workers are ready ! Starting data processing...")

        current_total = 0
        with tqdm(total=num_items, smoothing=0) as pbar:
            while True:
                new_total = sum([len(w) for w in self.workers])
                pbar.update(new_total - current_total)
                current_total = new_total
                sleep(1)
                if current_total == num_filepaths:
                    break

        logger.info("Finished data processing!")

    def _cached_list_filepaths(self, root: str) -> List[str]:
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
