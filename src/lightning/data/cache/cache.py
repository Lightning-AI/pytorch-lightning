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
import os
from typing import Any, Dict, Optional, Union

from lightning.data.cache.reader import BinaryReader
from lightning.data.cache.sampler import ChunkedIndex
from lightning.data.cache.writer import BinaryWriter
from lightning.data.datasets.env import _DistributedEnv

logger = logging.Logger(__name__)


class Cache:
    def __init__(
        self,
        cache_dir: str,
        remote_dir: Optional[str] = None,
        compression: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
    ):
        """The Cache enables to optimise dataset format for cloud training. This is done by grouping several elements
        together in order to accelerate fetching.

        Arguments:
            cache_dir: The path to where the chunks will be stored.
            remote_dir: The path to a remote folder where the data are located.
            compression: The name of the algorithm to reduce the size of the chunks.
            chunk_bytes: The maximum number of bytes within a chunk.
            chunk_size: The maximum number of items within a chunk.

        """
        super().__init__()
        self._writer = BinaryWriter(cache_dir, chunk_size=chunk_size, chunk_bytes=chunk_bytes, compression=compression)
        self._reader = BinaryReader(cache_dir, remote_dir=remote_dir, compression=compression)
        self._cache_dir = cache_dir
        self._is_done = False
        self._distributed_env = _DistributedEnv.detect()
        self._num_workers: Optional[int] = None

    def _setup(self, num_workers: int) -> None:
        """Called by the LightningDataLoader to ensure the num_workers is known."""
        self._num_workers = num_workers

    @property
    def filled(self) -> bool:
        """Returns whether the caching phase is done."""
        if self._num_workers is None:
            raise Exception("The Cache wasn't setup properly. HINT: Did you use the LightningDataLoader ?")
        if self._is_done:
            return True
        self._is_done = os.path.exists(os.path.join(self._cache_dir, "index.json"))
        return self._is_done

    def __setitem__(self, index, data) -> None:
        """Store an item in the writer."""
        self._writer[index] = data

    def __getitem__(self, index: Union[int, ChunkedIndex]) -> Dict[str, Any]:
        """Read an item in the reader."""
        if isinstance(index, int):
            index = ChunkedIndex(index, self._get_chunk_index_from_index(index))
        return self._reader.read(index)

    def done(self) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer.done()

    def merge(self, num_workers: int) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer.merge(num_workers)

    def __len__(self) -> int:
        return self._reader.get_length()

    def get_chunk_interval(self):
        return self._reader.get_chunk_interval()

    def _get_chunk_index_from_index(self, index: int) -> int:
        return self._reader._get_chunk_index_from_index(index)
