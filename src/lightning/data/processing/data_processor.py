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
from typing import Any, Dict, List, Optional, Union

from lightning import seed_everything
from lightning.data.constants import (
    _DEFAULT_FAST_DEV_RUN,
    _LIGHTNING_CLOUD_LATEST,
)
from lightning.data.processing.recipe import DataRecipe
from lightning.data.processing.strategy import _select_data_processor_strategy
from lightning.data.processing.strategy.queue import Queue
from lightning.data.processing.worker_functions import _download_data_target, _get_processor
from lightning.data.streaming.cache import Dir
from lightning.data.utilities.env import _cleanup_cache, _get_cache_data_dir, _get_fast_dev_run
from lightning.data.utilities.packing import _get_item_filesizes, _sort_greedily

if _LIGHTNING_CLOUD_LATEST:
    from lightning_cloud.resolver import _resolve_dir


logger = logging.Logger(__name__)


class DataProcessor:
    def __init__(
        self,
        input_dir: Union[str, Dir],
        output_dir: Optional[Union[str, Dir]] = None,
        num_workers: Optional[int] = None,
        num_downloaders: Optional[int] = None,
        num_uploaders: Optional[int] = None,
        delete_cached_files: bool = True,
        fast_dev_run: Optional[Union[bool, int]] = None,
        random_seed: Optional[int] = 42,
        reorder_files: bool = True,
    ):
        """The `DatasetOptimiser` provides an efficient way to process data across multiple machine into chunks to make
        training faster.

        Arguments:
            input_dir: The path to where the input data are stored.
            output_dir: The path to where the output data are stored.
            num_workers: The number of worker threads to use.
            num_downloaders: The number of file downloaders to use.
            num_uploaders: The number of file uploaders to use.
            delete_cached_files: Whether to delete the cached files.
            fast_dev_run: Whether to run a quick dev run.
            random_seed: The random seed to be set before shuffling the data.
            reorder_files: By default, reorders the files by file size to distribute work equally among all workers.
                Set this to ``False`` if the order in which samples are processed should be preserved.

        """
        self.strategy = _select_data_processor_strategy(input_dir, output_dir)

        print(f"Storing the files under {self.strategy.output_dir.path}")

        self.input_dir = _resolve_dir(input_dir)
        self.output_dir = _resolve_dir(output_dir)
        self.num_workers = num_workers or (1 if fast_dev_run else (os.cpu_count() or 1) * 4)
        self.num_downloaders = num_downloaders or 2
        self.num_uploaders = num_uploaders or 5
        self.delete_cached_files = delete_cached_files
        self.fast_dev_run = _get_fast_dev_run() if fast_dev_run is None else fast_dev_run
        self.workers: Any = []
        self.workers_tracker: Dict[int, int] = {}
        self.progress_queue: Optional[multiprocessing.Queue] = None
        self.error_queue: Queue = multiprocessing.Queue()
        self.stop_queues: List[multiprocessing.Queue] = []
        self.reorder_files = reorder_files
        self.random_seed = random_seed

        self.processors: List[multiprocessing.Process] = []
        self.dowloaders: List[multiprocessing.Process] = []
        self.uploaders: List[multiprocessing.Process] = []
        self.removers: List[multiprocessing.Process] = []

    def run(self, data_recipe: DataRecipe) -> None:
        """The `DataProcessor.run(...)` method triggers the data recipe processing over your dataset."""
        if not isinstance(data_recipe, DataRecipe):
            raise ValueError("The provided value should be a data recipe.")

        print(f"Setup started with fast_dev_run={self.fast_dev_run}.")

        self.before_setup()
        self.setup(data_recipe)
        self.dispatch(data_recipe)
        self.monitor()
        self.shutdown()

    def _exit_on_error(self, error: str) -> None:
        for w in self.workers:
            w.join(0)
        raise RuntimeError(f"We found the following error {error}.")

    def before_setup(self):
        # Force random seed to be fixed
        seed_everything(self.random_seed)

        # Register signal handler
        # signal.signal(signal.SIGINT, self._signal_handler)

        # Clean cache
        _cleanup_cache()

    def setup(self, data_recipe: DataRecipe) -> None:
        # Call the setup method of the user
        inputs: List[Any] = data_recipe.prepare_structure(
            self.strategy.input_dir.path if self.strategy.input_dir else None
        )

        if not isinstance(inputs, list):
            raise ValueError("The `prepare_structure` should return a list of item metadata.")

        if self.reorder_files:
            item_sizes = _get_item_filesizes(inputs, base_path=self.strategy.input_dir.path)
            inputs = _sort_greedily(inputs, item_sizes)

        if self.fast_dev_run:
            inputs_to_keep = self.fast_dev_run if type(self.fast_dev_run) is int else _DEFAULT_FAST_DEV_RUN
            inputs = inputs[:inputs_to_keep]
            print(f"Fast dev run is enabled. Limiting to {len(inputs)} inputs.")

        self.strategy.register_inputs(inputs)

    def dispatch(self, data_recipe: DataRecipe):
        self._dispatch_downloaders()
        self._dispatch_processors(data_recipe)

    def monitor(self):
        pass

    def shutdown(self):
        for downloader in self.dowloaders:
            downloader.join()

        for processor in self.processors:
            processor.join()

    def _dispatch_downloaders(self):
        self.ready_to_process_queue = multiprocessing.Queue()
        to_download_queue: Queue = self.strategy.get_global_queue()

        downloaders = []

        self.downloaders_event = multiprocessing.Event()

        for _ in range(self.num_downloaders):
            p = multiprocessing.Process(
                target=_download_data_target,
                args=(
                    self.event,
                    self.input_dir,
                    _get_cache_data_dir(),
                    to_download_queue,
                    self.ready_to_process_queue,
                ),
            )
            p.start()
            downloaders.append(p)

        self.dowloaders = downloaders

    def _dispatch_processors(self, data_recipe: DataRecipe) -> None:
        processors = []
        self.remove_queue = multiprocessing.Queue()

        self.processors_event = multiprocessing.Event()

        for worker_idx in range(self.num_workers):
            p = multiprocessing.Process(
                target=_get_processor(data_recipe),
                args=(
                    self.processors_event,
                    worker_idx,
                    self.num_workers,
                    self.input_dir,
                    self.output_dir,
                    _get_cache_data_dir(),
                    self.ready_to_process_queue,
                    self.remove_queue,
                ),
            )
            p.start()
            processors.append(p)

        self.processors = processors

    def _signal_handler(self, signal: Any, frame: Any) -> None:
        """On termination, we stop all the processes to avoid leaking RAM."""
        print("Not implemented")
