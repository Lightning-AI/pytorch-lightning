# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import multiprocessing
import os
import shutil
import tempfile
import types
from queue import Empty
from time import sleep
from typing import Any, Dict, List, Optional, Tuple
from urllib import parse

from lightning.data.constants import (
    _TORCH_GREATER_EQUAL_2_1_0,
)
from lightning.data.processing.recipe import DataChunkRecipe, DataRecipe, DataTransformRecipe
from lightning.data.processing.strategy.queue import Queue
from lightning.data.streaming import Cache
from lightning.data.streaming.cache import Dir
from lightning.data.streaming.client import S3Client
from lightning.data.utilities.env import _get_cache_data_dir, _get_cache_dir, _get_node_rank

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import tree_flatten, tree_unflatten

logger = logging.Logger(__name__)


def _wait_for_disk_usage_higher_than_threshold(input_dir: str, threshold_in_gb: int = 25, sleep_time: int = 3) -> None:
    usage = shutil.disk_usage(input_dir)

    while (usage.free / 1000 / 1000 / 1000) <= threshold_in_gb:
        sleep(sleep_time)
        usage = shutil.disk_usage(input_dir)

    return


def _download_data_target(
    event: multiprocessing.Event, input_dir: Dir, cache_dir: str, queue_in: Queue, queue_out: multiprocessing.Queue
) -> None:
    """This function is used to download data from a remote directory to a cache directory to optimise reading."""
    s3 = S3Client()

    while True:
        # 1. Exit if we have consumed all elements
        if event.is_set():
            return

        # 2. Fetch from the queue
        try:
            data: Dict[int, Any] = queue_in.get(timeout=1)
        except Empty:
            if event.is_set():
                return
            continue

        # We received signal that the last element has been consumed by one of the nodes.
        # Terminating this process
        if data is None:
            # Inform the other downloaders it is time to exit
            event.set()
            queue_out.put(None)
            return

        # We received no data, let's retry.
        if len(data) == 0:
            queue_out.put({})
            continue

        # Let's sort the data
        data = sorted([(key, item) for key, item in data.items()], key=lambda x: x[0])

        # Let's process the batch of data we received.
        for key, item in data:
            item, paths = _sanetize_item(item, input_dir, cache_dir)

            # 5. Check whether all the files are already downloaded
            if all(os.path.exists(p.replace(input_dir.path, cache_dir) if input_dir else p) for p in paths):
                queue_out.put({key: item})
                continue

            if input_dir.url is not None or input_dir.path is not None:
                if input_dir.url:
                    # 6. Wait for the removers to catch up when we are downloading data.
                    _wait_for_disk_usage_higher_than_threshold("/", 25)

                # 7. Download all the required paths to unblock the current index
                for path in paths:
                    local_path = path.replace(input_dir.path, cache_dir)

                    if input_dir.url:
                        path = path.replace(input_dir.path, input_dir.url)

                    obj = parse.urlparse(path)

                    if obj.scheme == "s3":
                        dirpath = os.path.dirname(local_path)

                        os.makedirs(dirpath, exist_ok=True)

                        with open(local_path, "wb") as f:
                            s3.client.download_fileobj(obj.netloc, obj.path.lstrip("/"), f)

                    elif os.path.isfile(path):
                        shutil.copyfile(path, local_path)
                    else:
                        raise ValueError(f"The provided {input_dir.url} isn't supported.")

            # 7. Inform the worker the current files are available
            queue_out.put({key: item})


def _sanetize_item(item: Any, input_dir: Dir, cache_dir: str) -> Tuple[Any, List[str]]:
    flattened_item, spec = tree_flatten(item)

    # For speed reasons, we assume starting with `input_dir` is enough to be a real file.
    # Other alternative would be too slow.
    # TODO: Try using dictionary for higher accurary.
    indexed_paths = {
        index: element
        for index, element in enumerate(flattened_item)
        if isinstance(element, str)
        and (
            element.startswith(input_dir.path) if input_dir is not None else os.path.exists(element)
        )  # For speed reasons
    }

    if len(indexed_paths) == 0:
        raise ValueError(f"The provided item {item} didn't contain any filepaths. The input_dir is {input_dir.path}.")

    paths = []
    for index, path in indexed_paths.items():
        paths.append(path)
        if input_dir:
            path = path.replace(input_dir.path, cache_dir)
        flattened_item[index] = path

    return tree_unflatten(flattened_item, spec), paths


def _remove_target(input_dir: Dir, cache_dir: str, queue_in: Queue) -> None:
    """This function is used to delete files from the cache directory to minimise disk space."""
    while True:
        # 1. Collect paths
        paths = queue_in.get()

        # 2. Terminate the process if we received a termination signal
        if paths is None:
            return

        # 3. Iterate through the paths and delete them sequentially.
        for path in paths:
            if input_dir:
                if not path.startswith(cache_dir):
                    path = path.replace(input_dir.path, cache_dir)

                if os.path.exists(path):
                    os.remove(path)

            elif os.path.exists(path) and "s3_connections" not in path:
                os.remove(path)


def _upload_fn(upload_queue: Queue, remove_queue: Queue, cache_dir: str, output_dir: Dir) -> None:
    """This function is used to upload optimised chunks from a local to remote dataset directory."""
    obj = parse.urlparse(output_dir.url if output_dir.url else output_dir.path)

    if obj.scheme == "s3":
        s3 = S3Client()

    while True:
        local_filepath: Optional[str] = upload_queue.get()

        # Terminate the process if we received a termination signal
        if local_filepath is None:
            return

        # Upload the file to the target cloud storage
        if not local_filepath.startswith(cache_dir):
            local_filepath = os.path.join(cache_dir, local_filepath)

        if obj.scheme == "s3":
            try:
                s3.client.upload_file(
                    local_filepath, obj.netloc, os.path.join(obj.path.lstrip("/"), os.path.basename(local_filepath))
                )
            except Exception as e:
                print(e)
        elif os.path.isdir(output_dir.path):
            shutil.copyfile(local_filepath, os.path.join(output_dir.path, os.path.basename(local_filepath)))
        else:
            raise ValueError(f"The provided {output_dir.path} isn't supported.")

        # Inform the remover to delete the file
        if remove_queue:
            remove_queue.put([local_filepath])


class ChunkProcessor:
    def __init__(self, data_recipe: DataRecipe, chunk_bytes: int, chunk_size: int, compression: str):
        self.data_recipe = data_recipe
        self.chunk_bytes = chunk_bytes
        self.chunk_size = chunk_size
        self.compression = compression

    def __call__(
        self,
        worker_index: int,
        num_workers: int,
        input_dir: Dir,
        cache_dir: str,
        ready_to_process_queue: multiprocessing.Queue,
        remove_queue: multiprocessing.Queue,
    ):
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.remove_queue = remove_queue

        self.counter = 0
        self.item_counter = 0

        self._set_environ_variables()
        self._create_cache()
        self._loop(ready_to_process_queue)

    def _set_environ_variables(self) -> None:
        # set the optimizer global rank and world_size
        os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = str(_get_node_rank() * self.num_workers + self.worker_index)
        os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(self.num_workers)

    def _create_cache(self) -> None:
        self.cache_data_dir = _get_cache_data_dir()
        os.makedirs(self.cache_data_dir, exist_ok=True)

        self.cache_chunks_dir = _get_cache_dir()
        os.makedirs(self.cache_chunks_dir, exist_ok=True)

        self.cache = Cache(
            self.cache_chunks_dir,
            chunk_bytes=self.chunk_bytes,
            chunk_size=self.chunk_size,
            compression=self.compression,
        )
        self.cache._reader._rank = _get_node_rank() * self.num_workers + self.worker_index

    def _loop(self, ready_to_process_queue):
        while True:
            data = ready_to_process_queue.get()

            if len(data) == 0:
                pass

            for index, item in data.items():
                self._process_item(item)

                self.item_counter += 1

                # if self.remove:
                #     self.remove_queue.put(self.paths[index])

                # try:
                #     self.stop_queue.get(timeout=0.0001)
                #     return
                # except Empty:
                #     pass

    def _process_item(self, item):
        try:
            self._current_item = item
            item_data_or_generator = self.data_recipe.prepare_item(self._current_item)
            if isinstance(item_data_or_generator, types.GeneratorType):
                for item_data in item_data_or_generator:
                    if item_data is not None:
                        chunk_filepath = self.cache._add_item(self.item_counter, item_data)
                        self._try_upload(chunk_filepath)
                        self.item_counter += 1
            elif item_data_or_generator is not None:
                chunk_filepath = self.cache._add_item(self.item_counter, item_data_or_generator)
                self._try_upload(chunk_filepath)
                self.item_counter += 1
        except Exception as e:
            raise RuntimeError(f"Failed processing {self._current_item}") from e

    def _try_upload(self, filepath: Optional[str]) -> None:
        if not filepath or (self.output_dir.url if self.output_dir.url else self.output_dir.path) is None:
            return

        assert os.path.exists(filepath), filepath
        self.to_upload_queues[self._counter % self.num_uploaders].put(filepath)

    @classmethod
    def from_data_recipe(cls, data_recipe: DataChunkRecipe):
        return cls(data_recipe, data_recipe.chunk_bytes, data_recipe.chunk_size, data_recipe.compression)


class TransformProcessor:
    def __init__(self, data_recipe: DataTransformRecipe):
        self.data_recipe = data_recipe

    def __call__(
        self,
        processors_event: multiprocessing.Event,
        worker_index: int,
        num_workers: int,
        input_dir: Dir,
        output_dir: Dir,
        cache_dir: str,
        ready_to_process_queue: multiprocessing.Queue,
        upload_queue: Optional[multiprocessing.Queue],
        remove_queue: Optional[multiprocessing.Queue],
    ):
        self.processors_event: multiprocessing.Event = processors_event
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.ready_to_process_queue = ready_to_process_queue
        self.upload_queue = upload_queue
        self.remove_queue = remove_queue
        self.counter = 0
        self.item_counter = 0

        self._set_environ_variables()
        self._loop()

    def _set_environ_variables(self) -> None:
        # set the optimizer global rank and world_size
        os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = str(_get_node_rank() * self.num_workers + self.worker_index)
        os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(self.num_workers)

    def _loop(self) -> None:
        while True:
            if self.processors_event.is_set():
                self._on_end()
                return

            try:
                data: Optional[Dict[int, Any]] = self.ready_to_process_queue.get(timeout=1)
            except Empty:
                if self.processors_event.is_set():
                    self._on_end()
                    return
                continue

            if data is None:
                # Inform all the processors that all the
                self.processors_event.set()
                self._on_end()
                return

            for _, item in data.items():
                self._process_item(item)
                self.item_counter += 1

    def _process_item(self, item):
        # Don't use a context manager to avoid deleting files that are being uploaded.
        output_dir = tempfile.mkdtemp()
        item_data = self.data_recipe.prepare_item(str(output_dir), item)
        if item_data is not None:
            raise ValueError(
                "When using a `DataTransformRecipe`, the `prepare_item` shouldn't return anything."
                " Simply store your files under the output_dir."
            )
        filepaths = []
        for directory, _, filenames in os.walk(output_dir):
            for filename in filenames:
                filepaths.append(os.path.join(directory, filename))

        if len(filepaths) == 0:
            raise RuntimeError("You haven't saved any files under the `output_dir`.")

        for filepath in filepaths:
            self._try_upload(filepath)

    def _try_upload(self, filepath: Optional[str]) -> None:
        if not filepath or (self.output_dir.url if self.output_dir.url else self.output_dir.path) is None:
            return

        assert os.path.exists(filepath), filepath
        self.upload_queue.put(filepath)

    def _on_end(self):
        print(f"Worker {str(_get_node_rank() * self.num_workers + self.worker_index)} is terminating.")

        if self.output_dir.url if self.output_dir.url else self.output_dir.path:
            self.upload_queue.put(None)

        if self.remove_queue:
            self.remove_queue.put(None)

        if self.progress_queue:
            self.progress_queue.put((self.worker_index, self._counter))

    @classmethod
    def from_data_recipe(cls, data_recipe: DataTransformRecipe):
        return cls(data_recipe)


def _get_processor(data_recipe: DataRecipe):
    if isinstance(data_recipe, DataChunkRecipe):
        return ChunkProcessor.from_data_recipe(data_recipe)
    return TransformProcessor.from_data_recipe(data_recipe)

    # class BaseWorker:
    #     def __init__(
    #         self,
    #         worker_index: int,
    #         num_workers: int,
    #         node_rank: int,
    #         data_recipe: "DataRecipe",
    #         input_dir: Dir,
    #         output_dir: Dir,
    #         items: List[Any],
    #         progress_queue: Queue,
    #         error_queue: Queue,
    #         stop_queue: Queue,
    #         num_downloaders: int,
    #         num_uploaders: int,
    #         remove: bool,
    #     ) -> None:
    #         """The BaseWorker is responsible to process the user data."""
    #         self.worker_index = worker_index
    #         self.num_workers = num_workers
    #         self.node_rank = node_rank
    #         self.data_recipe = data_recipe
    #         self.input_dir = input_dir
    #         self.output_dir = output_dir
    #         self.items = items
    #         self.num_items = len(self.items)
    #         self.num_downloaders = num_downloaders
    #         self.num_uploaders = num_uploaders
    #         self.remove = remove
    #         self.paths: List[List[str]] = []
    #         self.remover: Optional[Process] = None
    #         self.downloaders: List[Process] = []
    #         self.uploaders: List[Process] = []
    #         self.to_download_queues: List[Queue] = []
    #         self.to_upload_queues: List[Queue] = []
    #         self.stop_queue = stop_queue
    #         self.ready_to_process_queue: Queue = Queue()
    #         self.remove_queue: Queue = Queue()
    #         self.progress_queue: Queue = progress_queue
    #         self.error_queue: Queue = error_queue
    #         self._collected_items = 0
    #         self._counter = 0
    #         self._last_time = time()
    #         self._index_counter = 0
    #         self._current_item: Any = None

    #     def run(self) -> None:
    #         try:
    #             self._setup()
    #             self._loop()
    #         except Exception:
    #             traceback_format = traceback.format_exc()
    #             print(traceback_format)
    #             self.error_queue.put(traceback_format)
    #         print(f"Worker {str(_get_node_rank() * self.num_workers + self.worker_index)} is done.")

    #     def _setup(self) -> None:
    #         self._set_environ_variables()
    #         self._create_cache()
    #         self._collect_paths()
    #         self._start_downloaders()
    #         self._start_uploaders()
    #         self._start_remover()

    #     def _loop(self) -> None:
    #         num_downloader_finished = 0

    #         while True:
    #             index = self.ready_to_process_queue.get()

    #             if index is None:
    #                 num_downloader_finished += 1
    #                 if num_downloader_finished == self.num_downloaders:
    #                     print(f"Worker {str(_get_node_rank() * self.num_workers + self.worker_index)} is terminating.")

    #                     if isinstance(self.data_recipe, DataChunkRecipe):
    #                         self._handle_data_chunk_recipe_end()

    #                     if self.output_dir.url if self.output_dir.url else self.output_dir.path:
    #                         # Inform the uploaders they are doing working
    #                         for i in range(self.num_uploaders):
    #                             self.to_upload_queues[i].put(None)

    #                         # Wait for them all to be finished
    #                         for uploader in self.uploaders:
    #                             uploader.join()

    #                     if self.remove:
    #                         assert self.remover
    #                         self.remove_queue.put(None)
    #                         self.remover.join()

    #                     if self.progress_queue:
    #                         self.progress_queue.put((self.worker_index, self._counter))
    #                     return
    #                 continue

    #             if isinstance(self.data_recipe, DataChunkRecipe):
    #                 self._handle_data_chunk_recipe(index)
    #             else:
    #                 self._handle_data_transform_recipe(index)

    #             self._counter += 1

    #             # Don't send the last progress update, so the main thread awaits for the uploader and remover
    #             if self.progress_queue and (time() - self._last_time) > 1 and self._counter < (self.num_items - 2):
    #                 self.progress_queue.put((self.worker_index, self._counter))
    #                 self._last_time = time()

    #             if self.remove:
    #                 self.remove_queue.put(self.paths[index])

    #             try:
    #                 self.stop_queue.get(timeout=0.0001)
    #                 return
    #             except Empty:
    #                 pass

    #     def _set_environ_variables(self) -> None:
    #         # set the optimizer global rank and world_size
    #         os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = str(_get_node_rank() * self.num_workers + self.worker_index)
    #         os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(self.num_workers)

    #     def _create_cache(self) -> None:
    #         self.cache_data_dir = _get_cache_data_dir()
    #         os.makedirs(self.cache_data_dir, exist_ok=True)

    #         self.cache_chunks_dir = _get_cache_dir()
    #         os.makedirs(self.cache_chunks_dir, exist_ok=True)

    #         if isinstance(self.data_recipe, DataTransformRecipe):
    #             return

    #         self.cache = Cache(
    #             self.cache_chunks_dir,
    #             chunk_bytes=self.data_recipe.chunk_bytes,
    #             chunk_size=self.data_recipe.chunk_size,
    #             compression=self.data_recipe.compression,
    #         )
    #         self.cache._reader._rank = _get_node_rank() * self.num_workers + self.worker_index

    #     def _try_upload(self, filepath: Optional[str]) -> None:
    #         if not filepath or (self.output_dir.url if self.output_dir.url else self.output_dir.path) is None:
    #             return

    #         assert os.path.exists(filepath), filepath
    #         self.to_upload_queues[self._counter % self.num_uploaders].put(filepath)

    #     def _collect_paths(self) -> None:
    #         items = []
    #         for item in self.items:
    #             flattened_item, spec = tree_flatten(item)

    #             # For speed reasons, we assume starting with `self.input_dir` is enough to be a real file.
    #             # Other alternative would be too slow.
    #             # TODO: Try using dictionary for higher accurary.
    #             indexed_paths = {
    #                 index: element
    #                 for index, element in enumerate(flattened_item)
    #                 if isinstance(element, str)
    #                 and (
    #                     element.startswith(self.input_dir.path) if self.input_dir is not None else os.path.exists(element)
    #                 )  # For speed reasons
    #             }

    #             if len(indexed_paths) == 0:
    #                 raise ValueError(
    #                     f"The provided item {item} didn't contain any filepaths. The input_dir is {self.input_dir.path}."
    #                 )

    #             paths = []
    #             for index, path in indexed_paths.items():
    #                 paths.append(path)
    #                 if self.input_dir:
    #                     path = path.replace(self.input_dir.path, self.cache_data_dir)
    #                 flattened_item[index] = path

    #             self.paths.append(paths)

    #             items.append(tree_unflatten(flattened_item, spec))
    #             self._collected_items += 1

    #         self.items = items

    #     def _start_downloaders(self) -> None:
    #         for _ in range(self.num_downloaders):
    #             to_download_queue: Queue = Queue()
    #             p = Process(
    #                 target=_download_data_target,
    #                 args=(
    #                     self.input_dir,
    #                     self.cache_data_dir,
    #                     to_download_queue,
    #                     self.ready_to_process_queue,
    #                 ),
    #             )
    #             p.start()
    #             self.downloaders.append(p)
    #             self.to_download_queues.append(to_download_queue)

    #         for index, paths in enumerate(self.paths):
    #             self.to_download_queues[index % self.num_downloaders].put((index, paths))

    #         for downloader_index in range(self.num_downloaders):
    #             self.to_download_queues[downloader_index].put(None)

    #     def _start_remover(self) -> None:
    #         if not self.remove:
    #             return
    #         self.remover = Process(
    #             target=_remove_target,
    #             args=(
    #                 self.input_dir,
    #                 self.cache_data_dir,
    #                 self.remove_queue,
    #             ),
    #         )
    #         self.remover.start()

    #     def _start_uploaders(self) -> None:
    #         if self.output_dir.path is None and self.output_dir.url is None:
    #             return

    #         for _ in range(self.num_uploaders):
    #             to_upload_queue: Queue = Queue()
    #             p = Process(
    #                 target=_upload_fn,
    #                 args=(
    #                     to_upload_queue,
    #                     self.remove_queue,
    #                     self.cache_chunks_dir,
    #                     self.output_dir,
    #                 ),
    #             )
    #             p.start()
    #             self.uploaders.append(p)
    #             self.to_upload_queues.append(to_upload_queue)

    def _handle_data_chunk_recipe(self, index: int) -> None:
        try:
            self._current_item = self.items[index]
            item_data_or_generator = self.data_recipe.prepare_item(self._current_item)
            if isinstance(item_data_or_generator, types.GeneratorType):
                for item_data in item_data_or_generator:
                    if item_data is not None:
                        chunk_filepath = self.cache._add_item(self._index_counter, item_data)
                        self._try_upload(chunk_filepath)
                        self._index_counter += 1
            elif item_data_or_generator is not None:
                chunk_filepath = self.cache._add_item(self._index_counter, item_data_or_generator)
                self._try_upload(chunk_filepath)
                self._index_counter += 1
        except Exception as e:
            raise RuntimeError(f"Failed processing {self._current_item}") from e


#     def _handle_data_chunk_recipe_end(self) -> None:
#         chunks_filepaths = self.cache.done()

#         if chunks_filepaths and len(self.to_upload_queues):
#             for i, chunk_filepath in enumerate(chunks_filepaths):
#                 if isinstance(chunk_filepath, str) and os.path.exists(chunk_filepath):
#                     self.to_upload_queues[i % self.num_uploaders].put(chunk_filepath)

#     def _handle_data_transform_recipe(self, index: int) -> None:
#         # Don't use a context manager to avoid deleting files that are being uploaded.
#         output_dir = tempfile.mkdtemp()
#         item_data = self.data_recipe.prepare_item(str(output_dir), self.items[index])
#         if item_data is not None:
#             raise ValueError(
#                 "When using a `DataTransformRecipe`, the `prepare_item` shouldn't return anything."
#                 " Simply store your files under the output_dir."
#             )
#         filepaths = []
#         for directory, _, filenames in os.walk(output_dir):
#             for filename in filenames:
#                 filepaths.append(os.path.join(directory, filename))

#         if len(filepaths) == 0:
#             raise RuntimeError("You haven't saved any files under the `output_dir`.")

#         for filepath in filepaths:
#             self._try_upload(filepath)


# # class DataWorkerProcess(BaseWorker, Process):
# #     def __init__(self, *args: Any, **kwargs: Any) -> None:
# #         """The DataWorkerProcess is responsible to process the user data inside processes."""
# #         BaseWorker.__init__(self, *args, **kwargs)
# #         Process.__init__(self)
