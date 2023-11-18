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

from abc import ABC, abstractclassmethod, abstractmethod
from typing import Dict, TypeVar

from lightning_utilities.core.imports import RequirementCache, requires

_ZSTD_AVAILABLE = RequirementCache("zstd")

if _ZSTD_AVAILABLE:
    import zstd

TCompressor = TypeVar("TCompressor", bound="Compressor")


class Compressor(ABC):
    """Base class for compression algorithm."""

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        pass

    @abstractclassmethod
    def register(cls, compressors: Dict[str, "Compressor"]) -> None:
        pass


class ZSTDCompressor(Compressor):
    """Compressor for the zstd package."""

    @requires("zstd")
    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level
        self.extension = "zstd"

    @property
    def name(self) -> str:
        return f"{self.extension}:{self.level}"

    def compress(self, data: bytes) -> bytes:
        return zstd.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return zstd.decompress(data)

    @classmethod
    def register(cls, compressors: Dict[str, "Compressor"]) -> None:  # type: ignore
        if not _ZSTD_AVAILABLE:
            return

        # default
        compressors["zstd"] = ZSTDCompressor(4)

        for level in list(range(1, 23)):
            compressors[f"zstd:{level}"] = ZSTDCompressor(level)


_COMPRESSORS: Dict[str, Compressor] = {}

ZSTDCompressor.register(_COMPRESSORS)
