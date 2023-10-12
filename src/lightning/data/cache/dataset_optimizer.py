import logging
import os
import signal
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from threading import Thread
from time import sleep, time
from typing import Any, Callable, List, Literal, Optional, Tuple
from urllib import parse

import boto3
from tqdm import tqdm

from lightning import seed_everything
from lightning.data.cache import Cache
from lightning.data.cache.constants import _DEFAULT_FAST_DEV_RUN_ITEMS, _TORCH_2_1_0_AVAILABLE, _INDEX_FILENAME

if _TORCH_2_1_0_AVAILABLE:
    from torch.utils._pytree import tree_flatten, tree_unflatten

logger = logging.Logger(__name__)


def _get_num_nodes() -> int:
    """Returns the number of nodes."""
    return int(os.getenv("NUM_NODES", 1))


def _get_node_rank() -> int:
    """Returns the current node rank of the instance."""
    return int(os.getenv("NODE_RANK", 0))


def _get_fast_dev_mode() -> int:
    """Returns whether fast dev mode is enabled"""
    return bool(int(os.getenv("FAST_DEV_MODE", 1)))


def _download_data_target(src_dir: str, remote_src_dir: str, cache_dir: str, queue_in: Queue, queue_out: Queue) -> None:
    """This function is used to download data from a remote directory to a cache directory to optimise reading."""
    # 1. Create client
    s3 = boto3.client("s3")
    
    while True:
        # 2. Fetch from the queue
        r: Optional[Tuple[int, List[str]]] = queue_in.get()

        # 3. Terminate the process if we received a termination signal
        if r is None:
            queue_out.put(None)
            return

        # 4. Unpack
        index, paths = r

        # 5. Check whether all the files are already downloaded
        if all(os.path.exists(p.replace(src_dir, cache_dir)) for p in paths):
            queue_out.put(index)
            continue

        # 6. Download all the required paths to unblock the current index
        for path in paths:
            remote_path = path.replace(src_dir, remote_src_dir)
            obj = parse.urlparse(remote_path)

            if obj.scheme != "s3":
                raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_path}")

            local_path = path.replace(src_dir, cache_dir)

            dirpath = os.path.dirname(local_path)

            os.makedirs(dirpath, exist_ok=True)

            with open(local_path, "wb") as f:
                s3.download_fileobj(obj.netloc, obj.path.lstrip("/"), f)

        # 7. Inform the worker the current files are available
        queue_out.put(index)


def _remove_target(src_dir: str, cache_dir: str, queue_in: Queue) -> None:
    """This function is used to delete files from the cache directory to minimise disk space."""
    while True:
        # 1. Collect paths
        paths = queue_in.get()

        # 2. Terminate the process if we received a termination signal
        if paths is None:
            return

        # 3. Iterate through the paths and delete them sequentially.
        for path in paths:
            cached_filepath = path.replace(src_dir, cache_dir)

            if os.path.exists(cached_filepath):
                os.remove(cached_filepath)


def _upload_fn(upload_queue: Queue, remove_queue: Queue, cache_dir: str, remote_dst_dir: str) -> None:
    """This function is used to upload optimised chunks from a local to remote dataset directory."""
    obj = parse.urlparse(remote_dst_dir)

    if obj.scheme != "s3":
        raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_dst_dir}")

    s3 = boto3.client("s3")

    while True:
        r = upload_queue.get()

        # Terminate the process if we received a termination signal
        if r is None:
            return

        # Upload the file to the target cloud storage
        local_filepath = os.path.join(cache_dir, r)
        s3.upload_file(local_filepath, obj.netloc, os.path.join(obj.path.lstrip("/"), r))

        # Inform the remover to delete the file
        if remove_queue:
            remove_queue.put([local_filepath])
        


class BaseWorker:
    
    def __init__(
        self,
        worker_index: int,
        num_workers: int,
        start_index: int,
        dataset_name: str,
        node_rank: int,
        prepare_item: Callable,
        src_dir: str,
        remote_src_dir: str,
        remote_dst_dir: Optional[str],
        items: List[Any],
        worker_queue: Queue,
        error_queue: Queue,
        num_downloaders: int,
        remove: bool,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
        compression: Optional[str] = None,
    ):
        """The BaseWorker is responsible to process the user data."""
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.start_index = start_index
        self.dataset_name = dataset_name
        self.node_rank = node_rank
        self.prepare_item = prepare_item
        self.src_dir = src_dir
        self.remote_src_dir = remote_src_dir
        self.remote_dst_dir = remote_dst_dir
        self.items = items
        self.num_downloaders = num_downloaders
        self.remove = remove
        self.chunk_bytes = chunk_bytes
        self.chunk_size = chunk_size
        self.compression = compression
        self.paths = []
        self.remover = None
        self.downloaders = []
        self.to_download_queues = []
        self.ready_to_process_queue = Queue()
        self.remove_queue = Queue()
        self.upload_queue = Queue()
        self.worker_queue = worker_queue
        self.error_queue = error_queue
        self.uploader = None
        self._collected_items = 0
        self._counter = 0


    def run(self):
        try:
            self._setup()
            self._loop()
        except Exception:
            self.error_queue.put(traceback.format_exc())

    def _setup(self):
        self._set_environ_variables()
        self._create_cache()
        self._collect_paths()
        self._start_downloaders()
        self._start_uploader()
        self._start_remover()

    def _loop(self):
        
        num_downloader_finished = 0

        while True:
            r = self.ready_to_process_queue.get()
            
            if r is None:
                num_downloader_finished += 1
                if num_downloader_finished == self.num_downloaders:
                    self.remove_queue.put(None)
                    chunks_filepaths = self.cache.done()
                    
                    if chunks_filepaths:
                        for chunk_filepath in chunks_filepaths:
                            if isinstance(chunk_filepath, str) and os.path.exists(chunk_filepath):
                                self.upload_queue.put(chunk_filepath)

                    if self.remote_dst_dir:
                        self.upload_queue.put(None)
                        self.uploader.join()
                    return
                continue

            chunk_name = self.cache._add_item(r + self.start_index,
                self.prepare_item(self.items[r]) if self.prepare_item else self.items[r]
            )
            
            self._try_upload(chunk_name)
            
            self._counter += 1

            if self.worker_queue:
                self.worker_queue.put((self.worker_index, self._counter))

            if self.remove:
                self.remove_queue.put(self.paths[r])

    def _set_environ_variables(self):
        # set the optimizer global rank and world_size
        os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = str(_get_node_rank() * self.num_workers + self.worker_index)
        os.environ["DATA_OPTIMIZER_WORLD_SIZE"] = str(self.num_workers)

    def _create_cache(self):
        self.cache_chunks_dir = os.path.join("/cache", self.dataset_name)
        os.makedirs(self.cache_chunks_dir, exist_ok=True)

        self.cache = Cache(
            self.cache_chunks_dir, chunk_bytes=self.chunk_bytes, chunk_size=self.chunk_size, compression=self.compression
        )

        self.cache_data_dir = os.path.join("/cache", "data", self.dataset_name)
        os.makedirs(self.cache_data_dir, exist_ok=True)

    def _collect_paths(self):
        items = []
        for item in self.items:
            flattened_item, spec = tree_flatten(item)

            # For speed reasons, we assume starting with `self.src_dir` is enough to be a real file.
            # Other alternative would be too slow.
            # TODO: Try using dictionary for higher accurary.
            indexed_paths = {
                index: element
                for index, element in enumerate(flattened_item)
                if isinstance(element, str) and element.startswith(self.src_dir)  # For speed reasons
            }

            if len(indexed_paths) == 0:
                raise ValueError(f"The provided item {item} didn't contain any filepaths. {flattened_item}")

            paths = []
            for index, path in indexed_paths.items():
                tmp_path = path.replace(self.src_dir, self.cache_data_dir)
                flattened_item[index] = tmp_path
                paths.append(path)

            self.paths.append(paths)

            items.append(tree_unflatten(flattened_item, spec))

            self._collected_items += 1

        self.items = items

    def _start_downloaders(self):
        for _ in range(self.num_downloaders):
            to_download_queue = Queue()
            p = Process(
                target=_download_data_target,
                args=(
                    self.src_dir,
                    self.remote_src_dir,
                    self.cache_data_dir,
                    to_download_queue,
                    self.ready_to_process_queue,
                ),
            )
            p.start()
            self.downloaders.append(p)
            self.to_download_queues.append(to_download_queue)

        for index, paths in enumerate(self.paths):
            self.to_download_queues[index % self.num_downloaders].put((index, paths))

        for downloader_index in range(self.num_downloaders):
            self.to_download_queues[downloader_index].put(None)

    def _start_remover(self):
        if self.remove:
            self.remover = Process(
                target=_remove_target,
                args=(
                    self.src_dir,
                    self.cache_data_dir,
                    self.remove_queue,
                ),
            )
            self.remover.start()

    def _start_uploader(self):
        if self.remote_dst_dir:
            self.uploader = Process(
                target=_upload_fn,
                args=(
                    self.upload_queue,
                    self.remove_queue,
                    self.cache_chunks_dir,
                    self.remote_dst_dir,
                ),
            )
            self.uploader.start()


class DataWorkerThread(BaseWorker, Thread):
    def __init__(self, *args, **kwargs):
        """The DataWorkerThread is responsible to process the user data."""
        BaseWorker.__init__(self, *args, **kwargs)
        Thread.__init__(self, daemon=True)

    def join(self, timeout=None):
        for w in self.downloaders:
            w.kill()

        if self.remover is not None:
            self.remover.kill()

        super().join(timeout)

    def __len__(self):
        return self._counter

    @property
    def collected_items(self):
        return self._collected_items


class DataWorkerProcess(BaseWorker, Process):
    def __init__(self, *args, **kwargs):
        """The DataWorkerProcess is responsible to process the user data inside processes."""
        BaseWorker.__init__(self, *args, **kwargs)
        Process.__init__(self)


class WorkerType(Enum):
    THREAD = "thread"
    PROCESS = "process"


class DatasetOptimizer(ABC):
    @abstractmethod
    def prepare_dataset_structure(self, src_dir: str, filepaths: List[str]) -> List[Any]:
        """This function is meant to return a list of item metadata. Each item metadata should be enough to prepare a
        single item when called with the prepare_item.

        Example::

            # For a classification use case

            def prepare_dataset_structure(self, src_dir, filepaths)
                import numpy as np

                filepaths = ['class_a/file_1.ext', ..., 'class_b/file_1.ext', ...]
                classes = np.unique([filepath.split("/")[0] for filepath in filepaths])
                classes_to_idx_map = {c: idx for idx, c in enumerate(classes)}

                # Return pair with the filepath to the obj and its class
                # [('class_a/file_1.ext', 0), ... ('class_b/file_1.ext', 1)]
                return [(filepath, classes_to_idx_map[filepath.split("/")[0]]) for filepath in filepaths]

        Example::

            # For a image segmentation use case

            def prepare_dataset_structure(self, src_dir, filepaths)
                import numpy as np

                filepaths = ['file_1.JPEG', 'file_1.mask', .... 'file_N.JPEG', 'file_N.mask', ...]

                # [('file_1.JPEG', 'file_1.mask'), ... ('file_N.JPEG', 'file_N.mask')]
                return [(x[i], x[i+1]) for i in range(len(filepaths) -1)]

        """
        pass

    def prepare_item(self, metadata_item: Any) -> Any:
        """Using some metadata, prepare the associated item.

        The output of this function will be binarised

        """
        return metadata_item

    def __init__(
        self,
        name: str,
        src_dir: str,
        num_workers: Optional[int] = None,
        num_downloaders: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = 1 << 26,
        compression: Optional[str] = None,
        delete_cached_files: bool = True,
        src_resolver: Optional[Callable[[str], Optional[str]]] = None,
        worker_type: Literal["thread", "process"] = "process",
        fast_dev_run: Optional[bool] = None,
        remote_src_dir: Optional[str] = None,
        remote_dst_dir: Optional[str] = None,
        random_seed: Optional[int] = 42,
    ):
        """The `DatasetOptimiser` provides an efficient way to process data across multiple machine into chunks to make
        training faster.

        Arguments:
            name: The name of your dataset.
            src_dir: The path to where the data are stored.
            num_workers: The number of worker threads to use.
            num_downloaders: The number of file downloaders to use.
            chunk_size: The maximum number of elements to store within a chunk.
            chunk_bytes: The maximum number of bytes to store within a chunk.
            compression: The compression algorithm to apply on over the chunks.
            delete_cached_files: Whether to delete the cached files.
            fast_dev_run: Whether to run a quick dev run.
            remote_src_dir: The remote folder where the data are.
            remote_dst_dir: The remote folder where the optimised data will be stored.
            random_seed: The random seed to be set before shuffling the data.

        """
        self.name = name
        self.src_dir = src_dir
        self.num_workers = num_workers or (1 if fast_dev_run else os.cpu_count() * 4)
        self.num_downloaders = num_downloaders or (1 if fast_dev_run else 2)
        self.chunk_size = chunk_size
        self.chunk_bytes = chunk_bytes
        self.delete_cached_files = delete_cached_files
        self.compression = compression
        self.fast_dev_run = _get_fast_dev_mode() if fast_dev_run is None else fast_dev_run
        self.workers = []
        self.src_resolver = src_resolver or _LightningSrcResolver()
        self.dst_resolver = _LightningTargetResolver()
        self.worker_type = worker_type
        self.workers_tracker = {}
        self.worker_queue = None
        self.error_queue = Queue()
        self.remote_src_dir = (
            remote_src_dir
            if remote_src_dir is not None
            else (self.src_resolver(src_dir) if self.src_resolver else None)
        )
        self.remote_dst_dir = (
            remote_dst_dir if remote_dst_dir is not None else (self.dst_resolver(name) if self.dst_resolver else None)
        )
        self.random_seed = random_seed

    def run(self) -> None:
        """The `DatasetChunker.run(...)` method is used to trigger the data processing from your dataset into
        chunks."""
        t0 = time()
        print(f"Setup started for `{self.name}` with fast_dev_run={self.fast_dev_run}.")

        # Get the filepaths
        filepaths = self._cached_list_filepaths()

        if len(filepaths) == 0:
            raise RuntimeError(f"The provided directory {self.src_dir} is empty. ")

        # Force random seed to be fixed
        seed_everything(self.random_seed)

        # Call the setup method of the user
        user_items = self.prepare_dataset_structure(self.src_dir, filepaths)

        if not isinstance(user_items, list):
            raise ValueError("The setup_fn should return a list of item metadata.")

        # Associate the items to the workers based on num_nodes and node_rank
        begins, workers_user_items = self._associated_items_to_workers(user_items)
        print(f"Setup finished in {round(time() - t0, 3)} seconds. Found {len(user_items)} items to process.")

        if self.fast_dev_run:
            workers_user_items = [w[:_DEFAULT_FAST_DEV_RUN_ITEMS] for w in workers_user_items]
            print(f"Fast dev run is enabled. Limiting to {_DEFAULT_FAST_DEV_RUN_ITEMS} items per process.")

        num_items = sum([len(items) for items in workers_user_items])

        print(f"Starting {self.num_workers} workers")

        if self.remote_src_dir is None and self.src_resolver is not None:
            self.remote_src_dir = self.src_resolver(self.src_dir)
            print(f"The remote_dir is `{self.remote_src_dir}`.")

        signal.signal(signal.SIGINT, self._signal_handler)

        if self.worker_type == WorkerType.THREAD.value:
            self._create_thread_workers(user_items, begins, workers_user_items)
        else:
            self._create_process_workers(begins, workers_user_items)

        print("Workers are ready ! Starting data processing...")

        current_total = 0
        with tqdm(total=num_items, smoothing=0, position=-1, mininterval=1) as pbar:
            while True:
                if self.worker_type == WorkerType.THREAD.value:
                    new_total = sum([len(w) for w in self.workers])
                else:
                    try:
                        error = self.error_queue.get(timeout=0.001)
                        self._exit(error)
                    except Empty:
                        index, counter = self.worker_queue.get()
                        self.workers_tracker[index] = counter
                        new_total = sum(self.workers_tracker.values())
                    pbar.update(new_total - current_total)
                current_total = new_total
                if current_total >= num_items:
                    break

        if self.worker_type == WorkerType.THREAD.value:
            for w in self.workers:
                w.join(0)
        else:
            for w in self.workers:
                w.join(0)

        cache_dir = os.path.join("/cache", self.name)
        merge_cache = Cache(cache_dir, chunk_bytes=1)
        merge_cache.merge()
        self._upload_index(cache_dir)

        print("Finished data processing!")
        print()

    def _exit(self, error):
        if self.worker_type == WorkerType.THREAD.value:
            for w in self.workers:
                w.join(0)
        else:
            for w in self.workers:
                w.kill()
        raise RuntimeError(f"We found the following error {error}")

    def _create_thread_workers(self, user_items, begins, workers_user_items):
        num_items = len(user_items)
        current_total = 0
        with tqdm(total=num_items, smoothing=0) as pbar:
            for worker_idx, worker_user_items in enumerate(workers_user_items):
                new_total = sum([w.collected_items for w in self.workers])
                pbar.update(new_total - current_total)
                current_total = new_total
                worker = DataWorkerThread(
                    worker_idx,
                    self.num_workers,
                    begins[worker_idx],
                    self.name,
                    _get_node_rank(),
                    self.prepare_item,
                    self.src_dir,
                    self.remote_src_dir,
                    self.remote_dst_dir,
                    worker_user_items,
                    None,
                    self.error_queue,
                    self.num_downloaders,
                    self.delete_cached_files,
                    2 if self.fast_dev_run else self.chunk_size,  # In dev run, create chunks with 2 items
                    None if self.fast_dev_run else self.chunk_size,
                    self.compression,
                )
                worker.start()
                self.workers.append(worker)

            while True:
                new_total = sum([w.collected_items for w in self.workers])
                pbar.update(new_total - current_total)
                current_total = new_total
                sleep(1)
                if current_total == num_items:
                    break

    def _create_process_workers(self, begins, workers_user_items):
        self.worker_queue = Queue()
        for worker_idx, worker_user_items in enumerate(workers_user_items):
            worker = DataWorkerProcess(
                worker_idx,
                self.num_workers,
                begins[worker_idx],
                self.name,
                _get_node_rank(),
                self.prepare_item,
                self.src_dir,
                self.remote_src_dir,
                self.remote_dst_dir,
                worker_user_items,
                self.worker_queue,
                self.error_queue,
                self.num_downloaders,
                self.delete_cached_files,
                2 if self.fast_dev_run else self.chunk_size,  # In dev run, create chunks with 2 items
                None if self.fast_dev_run else self.chunk_size,
                self.compression,
            )
            worker.start()
            self.workers.append(worker)

    def _associated_items_to_workers(self, user_items: List[Any]):
        # Associate the items to the workers based on world_size and node_rank
        num_nodes = _get_num_nodes()
        current_node_rank = _get_node_rank()
        node_size = len(user_items) // num_nodes
        workers_user_items = []
        begins = []
        for node_rank in range(num_nodes):
            if node_rank != current_node_rank:
                continue
            is_last_node = node_rank == num_nodes - 1
            start_node = node_rank * node_size
            end_node = len(user_items) if is_last_node else (node_rank + 1) * node_size
            node_user_items = user_items[start_node:end_node]
            worker_size = len(node_user_items) // self.num_workers
            for worker_idx in range(self.num_workers):
                is_last = worker_idx == self.num_workers - 1
                begin = worker_idx * worker_size
                end = len(node_user_items) if is_last else (worker_idx + 1) * worker_size
                workers_user_items.append(user_items[begin:end])
                begins.append(begin)
            return begins, workers_user_items

    def _cached_list_filepaths(self) -> List[str]:
        home = os.path.expanduser("~")

        # TODO: Handle home directory in Jobs
        if home == "/home/zeus":
            home = "/teamspace/studios/this_studio"

        filepath = os.path.join(home, ".cache", f"{self.name}/filepaths.txt")

        if os.path.exists(filepath):
            lines = []
            with open(filepath) as f:
                for line in f.readlines():
                    lines.append(line.replace("\n", ""))
            return lines

        str(Path(self.src_dir).resolve())

        filepaths = []
        for dirpath, _, filenames in os.walk(self.src_dir):
            for filename in filenames:
                filepaths.append(os.path.join(dirpath, filename))

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            for filepath in filepaths:
                f.write(f"{filepath}\n")

        return filepaths

    def _signal_handler(self, signal, frame):
        for w in self.workers:
            if self.worker_type == WorkerType.THREAD.value:
                w.join(0)
            else:
                w.kill()
        os._exit(0)

    def _upload_index(self, cache_dir: str) -> None:
        if not self.remote_dst_dir:
            return

        obj = parse.urlparse(self.remote_dst_dir)
        s3 = boto3.client("s3")
        local_filepath = os.path.join(cache_dir, _INDEX_FILENAME)
        s3.upload_file(local_filepath, obj.netloc, os.path.join(obj.path.lstrip("/"), _INDEX_FILENAME))
