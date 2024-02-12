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

from lightning.data.constants import (
    _TORCH_DTYPES_MAPPING,
    _TORCH_GREATER_EQUAL_2_1_0,
)
from lightning.data.streaming.serializers import Serializer

if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import PyTree, tree_unflatten


class BaseItemLoader(ABC):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def setup(self, config: Dict, chunks: List, serializers: Dict[str, Serializer]) -> None:
        self._config = config
        self._chunks = chunks
        self._serializers = serializers

    def state_dict(self) -> Dict:
        return {}

    @abstractmethod
    def generate_intervals(self) -> List[Tuple[int, int]]:
        """Returns a list of tuple describing the indexes intervals of the chunks."""
        pass

    @abstractmethod
    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Logic to load the chunk in background to gain some time."""
        pass

    @abstractmethod
    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> Any:
        """Returns an item loaded from a chunk."""
        pass

    @abstractmethod
    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""
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

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        pass

    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> bytes:
        offset = (1 + (index - begin) if index >= begin else index + 1) * 4

        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        if chunk_filepath not in self._chunk_filepaths:
            exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0

            while not exists:
                sleep(0.1)
                exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0

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

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if os.path.exists(chunk_filepath):
            os.remove(chunk_filepath)


class TokensLoader(BaseItemLoader):
    def __init__(self, block_size: int):
        """The Tokens Loader is an optimizer item loader for NLP.

        Arguments:
            block_size: The context length to use during training.

        """

        super().__init__()
        self._block_size = block_size
        self._mmaps: Dict[int, np.memmap] = {}
        self._buffers: Dict[int, bytes] = {}
        self._dtype: Optional[torch.dtype] = None
        self._chunk_filepaths: Dict[str, bool] = {}

    def state_dict(self) -> Dict:
        return {
            "block_size": self._block_size,
        }

    def setup(self, config: Dict, chunks: List, serializers: Dict[str, Serializer]) -> None:
        super().setup(config, chunks, serializers)
        self._dtype = _TORCH_DTYPES_MAPPING[int(config["data_format"][0].split(":")[1])]
        if all(chunk["dim"] is None for chunk in self._chunks):
            raise ValueError("The provided chunks isn't properly setup.")

    def generate_intervals(self) -> List[Tuple[int, int]]:
        intervals = []
        begin = 0
        end = 0
        for chunk in self._chunks:
            dim = chunk["dim"]
            num_blocks = dim // self._block_size
            end += num_blocks
            intervals.append((begin, end))
            begin += num_blocks
        return intervals

    def _load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        if chunk_index in self._mmaps:
            return
        chunk = self._chunks[chunk_index]

        # Skip the header
        # The number of items + the number of offsets (number of items in the chunk + 1)
        # multiplied by the header encoding dtype (np.uint32)
        offset = (1 + chunk["chunk_size"] + 1) * 4
        mmap = np.memmap(chunk_filepath, mode="r", order="C", offset=offset)
        self._mmaps[chunk_index] = mmap
        self._buffers[chunk_index] = memoryview(mmap)  # type: ignore

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        # This is called within the prepare chunks thread, so we overlap data loading with data reading.
        if chunk_filepath not in self._chunk_filepaths:
            self._chunk_filepaths[chunk_filepath] = True

        if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0:
            self._load_chunk(chunk_index, chunk_filepath)

    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> torch.Tensor:
        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        if chunk_filepath not in self._chunk_filepaths:
            exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0

            while not exists:
                sleep(0.1)
                exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0

            self._chunk_filepaths[chunk_filepath] = True

        self._load_chunk(chunk_index, chunk_filepath)
        assert self._dtype

        buffer: bytes = self._buffers[chunk_index]
        offset = self._dtype.itemsize * (index - begin) * self._block_size
        return torch.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if os.path.exists(chunk_filepath):
            if chunk_index in self._buffers:
                del self._buffers[chunk_index]
            if chunk_index in self._mmaps:
                del self._mmaps[chunk_index]
            os.remove(chunk_filepath)
