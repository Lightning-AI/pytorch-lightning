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
from typing import Any, Callable, List, Literal, Optional
from urllib import parse

import boto3
from tqdm import tqdm

from lightning import seed_everything
from lightning.app.utilities.network import LightningClient
from lightning.data.cache import Cache
from lightning.data.cache.constants import _DEFAULT_FAST_DEV_RUN_ITEMS, _TORCH_2_1_0_AVAILABLE

if _TORCH_2_1_0_AVAILABLE:
    from torch.utils._pytree import tree_flatten, tree_unflatten

logger = logging.Logger(__name__)


class _Resolver(ABC):
    @abstractmethod
    def __call__(self, root: str) -> Optional[str]:
        pass


class _LightningSrcResolver(_Resolver):
    """The `_LightningSrcResolver` enables to retrieve a cloud storage path from a directory."""

    def __call__(self, root: str) -> Optional[str]:
        root_absolute = str(Path(root).absolute())

        if root_absolute.startswith("/teamspace/studios/this_studio"):
            return None

        if root_absolute.startswith("/.project/cloudspaces") and len(root_absolute.split("/")) > 3:
            return self._resolve_studio(root_absolute, None, root_absolute.split("/")[3])

        if root_absolute.startswith("/teamspace/studios") and len(root_absolute.split("/")) > 3:
            return self._resolve_studio(root_absolute, root_absolute.split("/")[3], None)

        if root_absolute.startswith("/teamspace/s3_connections") and len(root_absolute.split("/")) > 3:
            return self._resolve_s3_connections(root_absolute)

        return None

    def _resolve_studio(self, root: str, target_name: str, target_id: str) -> str:
        client = LightningClient()

        # Get the ids from env variables
        cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
        project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)

        if cluster_id is None:
            raise RuntimeError("The `cluster_id` couldn't be found from the environement variables.")

        if project_id is None:
            raise RuntimeError("The `project_id` couldn't be found from the environement variables.")

        clusters = client.cluster_service_list_clusters().clusters

        target_cloud_space = [
            cloudspace
            for cloudspace in client.cloud_space_service_list_cloud_spaces(
                project_id=project_id, cluster_id=cluster_id
            ).cloudspaces
            if cloudspace.name == target_name or cloudspace.id == target_id
        ]

        if not target_cloud_space:
            raise ValueError(f"We didn't find any matching Studio for the provided name `{target_name}`.")

        target_cluster = [cluster for cluster in clusters if cluster.id == target_cloud_space[0].cluster_id]

        if not target_cluster:
            raise ValueError(
                f"We didn't find a matching cluster associated with the id {target_cloud_space[0].cluster_id}."
            )

        bucket_name = target_cluster[0].spec.aws_v1.bucket_name

        return os.path.join(
            f"s3://{bucket_name}/projects/{project_id}/cloudspaces/{target_cloud_space[0].id}/code/content",
            *root.split("/")[4:],
        )

    def _resolve_s3_connections(self, root: str) -> str:
        client = LightningClient()

        # Get the ids from env variables
        project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
        if project_id is None:
            raise RuntimeError("The `project_id` couldn't be found from the environement variables.")

        target_name = root.split("/")[3]

        data_connections = client.data_connection_service_list_data_connections(project_id).data_connections

        data_connection = [dc for dc in data_connections if dc.name == target_name]

        if not data_connection:
            raise ValueError(f"We didn't find any matching data connection with the provided name `{target_name}`.")

        return os.path.join(data_connection[0].aws.source, *root.split("/")[4:])


class _LightningTargetResolver(_Resolver):
    """The `_LightningTargetResolver` generates a cloud storage path from a directory."""

    def __call__(self, name: str) -> Optional[str]:
        # Get the ids from env variables
        cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
        project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)

        if cluster_id is None or project_id is None:
            return

        client = LightningClient()

        clusters = client.cluster_service_list_clusters().clusters

        target_cluster = [cluster for cluster in clusters if cluster.id == cluster_id]

        if not target_cluster:
            raise ValueError(f"We didn't find a matching cluster associated with the id {cluster_id}.")

        prefix = os.path.join(f"projects/{project_id}/datasets/", name)

        import boto3

        s3 = boto3.client("s3")

        objects = s3.list_objects_v2(
            Bucket=target_cluster[0].spec.aws_v1.bucket_name,
            Delimiter="/",
            Prefix=prefix,
        )

        version = objects["KeyCount"] + 1 if objects["KeyCount"] else 0

        return os.path.join(f"s3://{target_cluster[0].spec.aws_v1.bucket_name}", prefix, f"version_{version}")


def _get_num_nodes() -> int:
    return int(os.getenv("NUM_NODES", 1))


def _get_node_rank() -> int:
    return int(os.getenv("NODE_RANK", 0))


def _get_fast_dev_mode() -> int:
    return bool(int(os.getenv("FAST_DEV_MODE", 1)))


def _download_data_target(src_dir: str, remote_src_dir: str, cache_dir: str, queue_in: Queue, queue_out: Queue) -> None:
    """This function is used to download data from a remote directory to a cache directory."""
    s3 = boto3.client("s3")
    while True:
        r = queue_in.get()

        if r is None:
            queue_out.put(None)
            return

        index, paths = r

        # Check whether all the files are already downloaded
        if all(os.path.exists(p.replace(src_dir, cache_dir)) for p in paths):
            queue_out.put(index)
            continue

        # Download all the required paths to unblock the current index
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

        queue_out.put(index)


def _remove_target(src_dir: str, cache_dir: str, queue_in: Queue) -> None:
    while True:
        paths = queue_in.get()

        if paths is None:
            return

        for path in paths:
            cached_filepath = path.replace(src_dir, cache_dir)

            if os.path.exists(cached_filepath):
                os.remove(cached_filepath)


class BaseWorker:
    def __init__(
        self,
        index: int,
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
        self.index = index
        self.num_workers = num_workers
        self.start_index = start_index
        self._dataset_name = dataset_name
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
        self._paths = []
        self._remover = None
        self._downloaders = []
        self._to_download_queues = []
        self._download_is_ready_queue = Queue()
        self._remove_queue = Queue()
        self._collected_items = 0
        self._counter = 0
        self._worker_queue = worker_queue
        self._error_queue = error_queue

    def run(self):
        try:
            # set the global rank
            os.environ["OPTIMIZER_GLOBAL_RANK"] = _get_node_rank() * self.num_workers + self.index
            self._create_cache()
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

                self.cache[r + self.start_index] = (
                    self.prepare_item(self.items[r]) if self.prepare_item else self.items[r]
                )

                self._counter += 1

                if self._worker_queue:
                    self._worker_queue.put((self.index, self._counter))

                if self.remove:
                    self._remove_queue.put(self._paths[r])
        except Exception:
            self._error_queue.put(traceback.format_exc())

    def _create_cache(self):
        cache_dir = os.path.join("/cache", self._dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

        self.cache = Cache(
            cache_dir, chunk_bytes=self.chunk_bytes, chunk_size=self.chunk_size, compression=self.compression
        )

        self.cache_dir = os.path.join("/cache", "data", self._dataset_name)
        os.makedirs(self.cache_dir, exist_ok=True)

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
                tmp_path = path.replace(self.src_dir, self.cache_dir)
                flattened_item[index] = tmp_path
                paths.append(path)

            self._paths.append(paths)

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
                    self.cache_dir,
                    to_download_queue,
                    self._download_is_ready_queue,
                ),
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
            self._remover = Process(
                target=_remove_target,
                args=(
                    self.src_dir,
                    self.cache_dir,
                    self._remove_queue,
                ),
            )
            self._remover.start()


class DataWorkerThread(BaseWorker, Thread):
    def __init__(self, *args, **kwargs):
        """The DataWorkerThread is responsible to process the user data."""
        BaseWorker.__init__(self, *args, **kwargs)
        Thread.__init__(self, daemon=True)

    def join(self, timeout=None):
        for w in self._downloaders:
            w.kill()

        if self._remover is not None:
            self._remover.kill()

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
                    self._get_node_rank(),
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
                self._get_node_rank(),
                self.prepare_item,
                self.src_dir,
                self.remote_src_dir,
                self.remote_dst_dir,
                worker_user_items,
                self.worker_queue,
                self.error_queue,
                self.num_downloaders,
                self.delete_cached_files,
                self.chunk_size,
                self.chunk_bytes,
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
