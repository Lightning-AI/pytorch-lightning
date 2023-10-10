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
from tqdm import tqdm

from lightning.app.utilities.network import LightningClient
from lightning.data.cache import Cache
from lightning.data.cache.constants import _TORCH_2_1_0_AVAILABLE
import signal

if _TORCH_2_1_0_AVAILABLE:
    from torch.utils._pytree import tree_flatten, tree_unflatten

logger = logging.Logger(__name__)


def _download_data_target(root: str, remote_root: str, cache_dir: str, queue_in: Queue, queue_out: Queue) -> None:
    """This function is used to download data from a remote directory to a cache directory."""
    s3 = boto3.client("s3")
    while True:
        r = queue_in.get()

        if r is None:
            queue_out.put(None)
            return

        index, paths = r

        # Check whether all the files are already downloaded
        if all(os.path.exists(p.replace(root, cache_dir)) for p in paths):
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


def _remove_target(root: str, cache_dir: str, queue_in: Queue) -> None:
    while True:
        paths = queue_in.get()

        if paths is None:
            return

        for path in paths:
            cached_filepath = path.replace(root, cache_dir)

            if os.path.exists(cached_filepath):
                os.remove(cached_filepath)


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
        self._remover = None
        self._downloaders = []
        self._to_download_queues = []
        self._download_is_ready_queue = Queue()
        self._remove_queue = Queue()
        self._collected_items = 0
        self._counter = 0

        self._create_cache()

    def join(self, timeout=None):
        for w in self._downloaders:
            w.kill()

        if self._remover:
            self._remover.kill()

        super().join(timeout)

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
                    self._remove_queue.put(None)
                    self.cache.done()
                    return
                continue

            self.cache[r + self.start_index] = self.prepare_item(self.items[r]) if self.prepare_item else self.items[r]

            with self._lock:
                self._counter += 1

            if self.remove:
                self._remove_queue.put(self._paths[r])

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

            self._paths.append(paths)

            items.append(tree_unflatten(flattened_item, spec))

            with self._lock:
                self._collected_items += 1

        self.items = items

    def _start_downloaders(self):
        for _ in range(self.num_downloaders):
            to_download_queue = Queue()
            p = Process(
                target=_download_data_target,
                args=(self.root, self.remote_root, self.cache_dir, to_download_queue, self._download_is_ready_queue),
            )
            p.start()
            self._downloaders.append(p)
            self._to_download_queues.append(to_download_queue)

        for index, paths in enumerate(self._paths):
            self._to_download_queues[index % self.num_downloaders].put((index, paths))

        for downloader_index in range(self.num_downloaders):
            self._to_download_queues[downloader_index].put(None)

    def _start_remover(self):
        if self.remove:
            self._remover = Process(target=_remove_target, args=(self.root, self.cache_dir, self._remove_queue,))
            self._remover.start()


class DataProcessor:
    def __init__(
        self,
        setup: Callable[[str, str], List[str]],
        prepare_item: Optional[Callable] = None,
        num_workers: int = os.cpu_count() * 4,
        num_downloaders: int = 1,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = 1 << 26,
        compression: Optional[str] = None,
        delete_cached_files: bool = True,
        resolver: Optional[Callable[[str], str]] = None,
    ):
        """The `DataProcessor` provides an efficient way to process data across multiple nodes in the cloud into
        chunks.

        Arguments:
            setup: The function used to organize the dataset metadata.
            prepare_item: The function to used to prepare a single item from its metadata, including filepaths.
            num_workers: The number of worker threads to use.
            num_downloaders: The number of file downloaders to use.
            chunk_size: The maximum number of elements to store within a chunk.
            chunk_bytes: The maximum number of bytes to store within a chunk.
            compression: The compression algorithm to apply on over the chunks.
            delete_cached_files: Whether to delete the cached files.

        """
        self.setup = setup
        self.prepare_item = prepare_item
        self.num_workers = num_workers
        self.num_downloaders = num_downloaders
        self.chunk_size = chunk_size
        self.chunk_bytes = chunk_bytes
        self.delete_cached_files = delete_cached_files
        self.compression = compression
        self.workers = []
        self.resolver = resolver

    def run(self, root: str, remote_root: Optional[str] = None) -> None:
        t0 = time()
        logger.info("Setup started")

        world_size = self._get_world_size()
        node_rank = self._get_node_rank()

        # Get the filepaths
        root = str(Path(root).resolve())
        filepaths = self._cached_list_filepaths(root)
        num_filepaths = len(filepaths)

        # Call the setup method of the user
        user_items = self.setup(root, filepaths)

        # Associate the items to the workers based on world_size and node_rank
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

        if remote_root is None and self.resolver is not None:
            remote_root = self.resolver(root)

        signal.signal(signal.SIGINT, self._signal_handler)

        num_items = len(user_items)
        current_total = 0
        with tqdm(total=num_items, smoothing=0) as pbar:
            for worker_idx, worker_user_items in enumerate(workers_user_items):
                new_total = sum([w.collected_items for w in self.workers])
                pbar.update(new_total - current_total)
                current_total = new_total
                worker = DataWorker(
                    worker_idx,
                    begins[worker_idx],
                    self.prepare_item,
                    root,
                    remote_root,
                    worker_user_items.tolist(),
                    self.num_downloaders,
                    self.delete_cached_files,
                    self.chunk_size,
                    self.chunk_bytes,
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

        logger.info("Workers are ready ! Starting data processing...")

        current_total = 0
        with tqdm(total=num_items, smoothing=0) as pbar:
            while True:
                new_total = sum([len(w) for w in self.workers])
                pbar.update(new_total - current_total)
                current_total = new_total
                sleep(1)
                if current_total == num_items:
                    break

        for w in self.workers:
            w.join(0)

        logger.info("Finished data processing!")

    def _cached_list_filepaths(self, root: str) -> List[str]:
        algo = hashlib.new("sha256")
        algo.update(root.encode("utf-8"))
        root_hash = algo.hexdigest()

        filepath = f"/cache/{root_hash}/filepaths.txt"

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

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            for filepath in filepaths:
                f.write(f"{filepath}\n")

        return filepaths

    def _signal_handler(signal, frame):
        for w in self.workers:
            w.join(0)
        sys.exit(0)

    def _get_world_size(self) -> int:
        return os.getenv('WORLD_SIZE', 1)

    def _get_node_rank(self) -> int:
        return os.getenv('NODE_RANK', 1)


class LightningResolver:
    def __call__(self, root: str) -> str:
        if root.startswith("/teamspace/studios") and not root.startswith("/teamspace/studios/this_studio"):
            return self._resolve_studio(root)

        if root.startswith("/teamspace/s3_connections"):
            return self._resolve_s3_connections(root)

    def _resolve_studio(self, root: str) -> str:
        client = LightningClient()

        # Get the ids from env variables
        cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
        project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
        cloud_space_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID", None)
        
        target_name = root.split("/")[3]

        clusters = client.cluster_service_list_clusters().clusters

        target_cloud_space = [cloudspace 
            for cloudspace in client.cloud_space_service_list_cloud_spaces(project_id=project_id, cluster_id=cluster_id).cloudspaces
            if cloudspace.name == root.split("/")[3]]

        if not target_cloud_space:
            raise ValueError(f"We didn't find a matching Studio for the provided {root}.")

        target_cluster = [cluster for cluster in clusters if cluster.id == target_cloud_space[0].cluster_id]

        if not target_cluster:
            raise ValueError(f"We didn't find a matching cluster for the provided {root}.")

        bucket_name = target_cluster[0].spec.aws_v1.bucket_name

        return os.path.join(f"s3://{bucket_name}/projects/{project_id}/cloudspaces/{target_cloud_space[0].id}/code/content", *root.split("/")[4:])

    def _resolve_s3_connections(self, root: str) -> str:
        raise NotImplementedError
