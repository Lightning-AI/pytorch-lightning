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
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from lightning.data.builder.compression import _COMPRESSORS


class BaseWriter(ABC):
    def __init__(
        self,
        out_dir: str,
        chunk_size: int = 1 << 26,
        compression: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self._out_dir = out_dir

        if not os.path.exists(self._out_dir):
            raise Exception(f"The provided output directory {self._out_dir} doesn't exists.")

        self._chunk_size = chunk_size
        self._compression = compression
        self._name = name

        if compression and compression not in _COMPRESSORS:
            raise Exception(f"The provided compression {compression} isn't available in {sorted(_COMPRESSORS)}")

        self._compressor = _COMPRESSORS[compression]

        self._current_chunk_size = 0
        self._counter = 0
        self._serialized_items = []
        self._serializers: List[Serializer] = []
        self._chunks = []

    @property
    def is_cached(self) -> bool:
        return os.path.exists(os.path.join(self._out_dir, "index.json"))

    def get_config(self) -> Dict[str, Any]:
        return {"compression": self._compression, "chunk_size": self._chunk_size}

    @property
    def available_serializers(self):
        return self._serializers

    @abstractmethod
    def serialize(self, data: any) -> bytes:
        """Convert a given data type into its bytes format."""

    @abstractmethod
    def write_chunk(self, rank: int) -> None:
        """Write the current chunk to the filesystem."""

    def reset(self) -> None:
        """Reset the writer to handle the next chunk."""
        self._serialized_items = []
        self._current_chunk_size = 0

    def write(self, items: any, rank):
        serialized_items = self.serialize(items)
        serialized_items_size = len(serialized_items)

        if self._chunk_size < self._current_chunk_size + serialized_items_size:
            self.write_chunk(rank)
            self.reset()
            self._counter += 1

        self._serialized_items.append(serialized_items)
        self._current_chunk_size += serialized_items_size

    def write_file(
        self,
        raw_data: bytes,
        filename: str,
    ) -> None:
        if self._compression:
            raw_data = self._compressor.compress(raw_data)
        filepath = os.path.join(self._out_dir, filename)
        with open(filepath, "wb") as out:
            out.write(raw_data)

    def write_chunks_index(self, rank: int):
        filepath = os.path.join(self._out_dir, f"{rank}.index.json")
        with open(filepath, "w") as out:
            json.dump({"chunks": self._chunks}, out, sort_keys=True)

    def done(self, rank: int):
        if self._serialized_items:
            self.write_chunk(rank)
            self.write_chunks_index(rank)
            self.reset()


class Serializer(ABC):
    @abstractmethod
    def serialize(self, data: any) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> any:
        pass
