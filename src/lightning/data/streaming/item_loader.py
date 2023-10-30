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
from abc import ABC, abstractmethod
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from lightning.data.streaming.constants import (
    _TORCH_DTYPES_MAPPING,
    _TORCH_GREATER_EQUAL_2_1_0,
)
from lightning.data.streaming.serializers import _SERIALIZERS

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import PyTree, tree_unflatten


class BaseItemLoader(ABC):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def setup(self, config: Dict, chunks: List) -> None:
        self._config = config
        self._chunks = chunks
        self._serializers = _SERIALIZERS

    @abstractmethod
    def generate_intervals(self) -> List[Tuple[int, int]]:
        """Returns a list of tuple describing the indexes intervals of the chunks."""
        pass

    @abstractmethod
    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> Any:
        """Returns an item loaded from a chunk."""
        pass


class PyTreeLoader(BaseItemLoader):
    """The Pytree Loader is the default loader of the Cache object."""

    def __init__(self) -> None:
        self._chunk_filepaths: Dict[str, bool] = {}

    def generate_intervals(self) -> List[Tuple[int, int]]:
        intervals = []
        begin = 0
        end = 0
        for chunk in self._chunks:
            end += chunk["chunk_size"]
            intervals.append((begin, end))
            begin += chunk["chunk_size"]
        return intervals

    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> bytes:
        offset = (1 + (index - begin) if index >= begin else index + 1) * 4

        if chunk_filepath not in self._chunk_filepaths:
            while not os.path.exists(chunk_filepath):
                sleep(0.001)
            self._chunk_filepaths[chunk_filepath] = True

        with open(chunk_filepath, "rb", 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
            fp.seek(begin)
            data = fp.read(end - begin)
        return self.deserialize(data)

    def deserialize(self, raw_item_data: bytes) -> "PyTree":
        """Deserialize the raw bytes into their python equivalent."""
        idx = len(self._config["data_format"]) * 4
        sizes = np.frombuffer(raw_item_data[:idx], np.uint32)
        data = []
        for size, data_format in zip(sizes, self._config["data_format"]):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config["data_spec"])


class TokensLoader(BaseItemLoader):
    def __init__(self, block_size: int):
        """The Tokens Loader is an optimizer item loader for NLP.

        Arguments:
            block_size: The context length to use during training.

        """

        super().__init__()
        self._block_size = block_size
        self._intervals: List[Tuple[int, int]] = []
        self._mmaps: Dict[int, np.memmap] = {}
        self._buffers: Dict[int, bytes] = {}
        self._dtype: Optional[torch.dtype] = None

    def setup(self, config: Dict, chunks: List) -> None:
        super().setup(config, chunks)
        self._dtype = _TORCH_DTYPES_MAPPING[int(config["data_format"][0].split(":")[1])]
        if all(chunk["dim"] is None for chunk in self._chunks):
            raise ValueError("The provided chunks isn't properly setup.")

    def generate_intervals(self) -> List[Tuple[int, int]]:
        begin = 0
        end = 0
        for chunk in self._chunks:
            dim = chunk["dim"]
            num_blocks = dim // self._block_size
            end += num_blocks
            self._intervals.append((begin, end))
            begin += num_blocks

        return self._intervals

    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> torch.Tensor:
        while not os.path.exists(chunk_filepath):
            sleep(0.0001)

        if chunk_index not in self._mmaps:
            # TODO: Add deletion and memmap close
            chunk = self._chunks[chunk_index]
            offset = (1 + chunk["chunk_size"] + 1) * 4
            mmap = np.memmap(chunk_filepath, mode="r", order="C", offset=offset)
            self._mmaps[chunk_index] = mmap
            self._buffers[chunk_index] = memoryview(mmap)  # type: ignore

        assert self._dtype

        buffer: bytes = self._buffers[chunk_index]
        offset = self._dtype.itemsize * ((index - begin) if index >= begin else index + 1)
        return torch.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
