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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from lightning.data.datasets.env import _DistributedEnv
from lightning.data.streaming.constants import (
    _INDEX_FILENAME,
    _LIGHTNING_CLOUD_GREATER_EQUAL_0_5_50,
    _TORCH_GREATER_EQUAL_2_1_0,
)
from lightning.data.streaming.item_loader import BaseItemLoader
from lightning.data.streaming.reader import BinaryReader
from lightning.data.streaming.sampler import ChunkedIndex
from lightning.data.streaming.writer import BinaryWriter
from lightning.data.utilities.format import _convert_bytes_to_int

logger = logging.Logger(__name__)

if _LIGHTNING_CLOUD_GREATER_EQUAL_0_5_50:
    from lightning_cloud.resolver import _resolve_dir


@dataclass
class Dir:
    """Holds a directory path and possibly its associated remote URL."""

    path: str
    url: Optional[str] = None


class Cache:
    def __init__(
        self,
        input_dir: Optional[Union[str, Dir]],
        compression: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[Union[int, str]] = None,
        item_loader: Optional[BaseItemLoader] = None,
        max_cache_size: Union[int, str] = "200GB",
    ):
        """The Cache enables to optimise dataset format for cloud training. This is done by grouping several elements
        together in order to accelerate fetching.

        Arguments:
            input_dir: The path to where the chunks will be or are stored.
            name: The name of dataset in the cloud.
            version: The version of the dataset in the cloud to use. By default, we will use the latest.
            compression: The name of the algorithm to reduce the size of the chunks.
            chunk_bytes: The maximum number of bytes within a chunk.
            chunk_size: The maximum number of items within a chunk.
            item_loader: The object responsible to generate the chunk intervals and load an item froma chunk.
            max_cache_size: The maximum cache size used by the reader when fetching the chunks.

        """
        super().__init__()
        if not _TORCH_GREATER_EQUAL_2_1_0:
            raise ModuleNotFoundError("PyTorch version 2.1 or higher is required to use the cache.")

        if not _LIGHTNING_CLOUD_GREATER_EQUAL_0_5_50:
            raise ModuleNotFoundError("Lightning Cloud 0.5.50 or higher is required to use the cache.")

        input_dir = _resolve_dir(input_dir)
        self._cache_dir = input_dir.path
        self._writer = BinaryWriter(
            self._cache_dir, chunk_size=chunk_size, chunk_bytes=chunk_bytes, compression=compression
        )
        self._reader = BinaryReader(
            self._cache_dir,
            max_cache_size=_convert_bytes_to_int(max_cache_size) if isinstance(max_cache_size, str) else max_cache_size,
            remote_input_dir=input_dir.url,
            compression=compression,
            item_loader=item_loader,
        )
        self._is_done = False
        self._distributed_env = _DistributedEnv.detect()

    @property
    def filled(self) -> bool:
        """Returns whether the caching phase is done."""
        if self._is_done:
            return True
        self._is_done = os.path.exists(os.path.join(self._cache_dir, _INDEX_FILENAME))
        return self._is_done

    def __setitem__(self, index: int, data: Any) -> None:
        """Store an item in the writer."""
        self._writer[index] = data

    def _add_item(self, index: int, data: Any) -> Optional[str]:
        """Store an item in the writer and optionally return the chunk path."""
        return self._writer.add_item(index, data)

    def __getitem__(self, index: Union[int, ChunkedIndex]) -> Dict[str, Any]:
        """Read an item in the reader."""
        if isinstance(index, int):
            index = ChunkedIndex(index, self._get_chunk_index_from_index(index))
        return self._reader.read(index)

    def done(self) -> Optional[List[str]]:
        """Inform the writer the chunking phase is finished."""
        return self._writer.done()

    def merge(self, num_workers: int = 1, node_rank: Optional[int] = None) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer.merge(num_workers, node_rank=node_rank)

    def _merge_no_wait(self, node_rank: Optional[int] = None) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer._merge_no_wait(node_rank=node_rank)

    def __len__(self) -> int:
        return self._reader.get_length()

    def get_chunk_intervals(self) -> List[Tuple[int, int]]:
        return self._reader.get_chunk_intervals()

    def _get_chunk_index_from_index(self, index: int) -> int:
        return self._reader._get_chunk_index_from_index(index)
