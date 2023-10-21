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

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from lightning.data.streaming.constants import _INDEX_FILENAME, _TORCH_GREATER_EQUAL_2_1_0
from lightning.data.streaming.downloader import get_downloader_cls
from lightning.data.streaming.item_loader import BaseItemLoader, PyTreeLoader, TokensLoader
from lightning.data.streaming.sampler import ChunkedIndex

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import treespec_loads


class ChunksConfig:
    def __init__(self, cache_dir: str, remote_dir: Optional[str], item_loader: Optional[BaseItemLoader] = None) -> None:
        """The ChunksConfig reads the index files associated a chunked dataset and enables to map an index to its
        chunk.

        Arguments:
            cache_dir: The path to cache folder.
            remote_dir: The path to a remote folder where the data are located.
                The scheme needs to be added to the path.

        """
        self._cache_dir = cache_dir
        self._intervals: List[Tuple[int, int]] = []
        self._config = None
        self._chunks = []
        self._remote_dir = remote_dir
        self._item_loader = item_loader or PyTreeLoader()

        with open(os.path.join(self._cache_dir, _INDEX_FILENAME)) as f:
            data = json.load(f)
            self._config = data["config"]
            self._validate_item_loader()
            self._chunks.extend(data["chunks"])

        self._config["data_spec"] = treespec_loads(self._config["data_spec"])

        self._item_loader.setup(self._config, self._chunks)
        self._intervals = self._item_loader.generate_intervals()
        self._length = self._intervals[-1][-1]
        self._downloader = None

        if remote_dir:
            self._downloader = get_downloader_cls(remote_dir)(remote_dir, cache_dir, self._chunks)

    def download_chunk_from_index(self, chunk_index: int) -> None:
        chunk_filename = self._chunks[chunk_index]["filename"]

        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)

        if os.path.exists(local_chunkpath):
            return

        if self._downloader is None:
            raise RuntimeError("The downloader should be defined.")

        self._downloader.download_chunk_from_index(chunk_index)

    @property
    def intervals(self) -> List[Tuple[int, int]]:
        if self._intervals is None:
            raise RuntimeError("The intervals should be defined.")
        return self._intervals

    @property
    def data_format(self) -> Any:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config["data_format"]

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config

    def _get_chunk_index_from_index(self, index: int) -> int:
        for chunk_index, internal in enumerate(self._intervals):
            if internal[0] <= index < internal[1]:
                return chunk_index
        raise ValueError(
            f"The provided index {index} didn't find a match within the chunk intervals {self._intervals}."
        )

    def __getitem__(self, index: ChunkedIndex) -> Tuple[str, int, int]:
        """Find the associated chunk metadata."""
        chunk = self._chunks[index.chunk_index]
        return os.path.join(self._cache_dir, chunk["filename"]), *self._intervals[index.chunk_index]

    @classmethod
    def load(
        cls, cache_dir: str, remote_dir: Optional[str] = None, item_loader: Optional[BaseItemLoader] = None
    ) -> Optional["ChunksConfig"]:
        cache_index_filepath = os.path.join(cache_dir, _INDEX_FILENAME)

        if isinstance(remote_dir, str):
            downloader = get_downloader_cls(remote_dir)(remote_dir, cache_dir, [])
            downloader.download_file(os.path.join(remote_dir, _INDEX_FILENAME), cache_index_filepath)

        if not os.path.exists(cache_index_filepath):
            return None

        return ChunksConfig(cache_dir, remote_dir, item_loader)

    def __len__(self) -> int:
        return self._length

    def _validate_item_loader(self) -> None:
        assert self._config
        if (
            len(self._config["data_format"]) == 1
            and self._config["data_format"][0].startswith("no_header_tensor")
            and not isinstance(self._item_loader, TokensLoader)
        ):
            raise ValueError("Please, use Cache(..., item_loader=TokensLoader(block_size=...))")
