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

import os
from typing import Union, Dict, Optional
from lightning.data.builder.writer import Writer
from lightning.data.builder.reader import Reader
from torch.utils.data import get_worker_info
import torch
from torch.distributed import is_initialized, is_available, get_world_size 
from lightning.data.datasets.env import _WorkerEnv, _DistributedEnv
from torch.utils.data.dataloader import DataLoader, _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter

class Cache:

    def __init__(self, cache_dir: str, data_format: Union[Dict[str, any], str], compression: Optional[str] = None, chunk_size: int = 2 << 26):
        super().__init__()
        self._writer = Writer(cache_dir, data_format, chunk_size)
        self._reader = Reader(cache_dir)
        self._cache_dir = cache_dir

        self._env = _DistributedEnv.detect()
        self._worker_env = None
        self._rank = None

    @property
    def rank(self):
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    @property
    def filled(self) -> bool:
        files = os.listdir(self._cache_dir)
        return any(f.endswith("index.json") for f in files)

    def __setitem__(self, index, data):
        self._writer.write(data, self.rank)

    def __getitem__(self, index):
        self._reader.read(index, self.rank)

    def done(self):
        self._writer.done(self.rank)


class _SingleProcessDataLoaderIterPatch(_SingleProcessDataLoaderIter):

    def _next_data(self):
        try:
            return super()._next_data()
        except StopIteration:
            for v in self._dataset_fetcher.dataset.__dict__.values():
                if isinstance(v, Cache):
                    v.done()
            raise StopIteration()

class CacheDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterPatch(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)
