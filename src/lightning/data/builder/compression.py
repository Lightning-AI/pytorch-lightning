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

from typing import Dict, TypeVar
from abc import ABC, abstractmethod, abstractclassmethod
import zstd

TCompressor = TypeVar("TCompressor", bound="Compressor")

class Compressor(ABC):

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        pass

    @abstractclassmethod
    def register(cls, compressors: Dict[str, TCompressor]):
        pass


class ZSTDCompressor(Compressor):

    def __init__(self, level):
        super().__init__()
        self.level = level
        self.extension = 'zstd'

    @property
    def name(self):
        return f"{self.extension}:{self.level}"

    def compress(self, data: bytes) -> bytes:
        return zstd.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return zstd.decompress(data)

    @classmethod
    def register(cls,  compressors):
        # default
        compressors["zstd"] = ZSTDCompressor(4)

        for level in list(range(1, 23)):
            compressors[f"zstd:{level}"] = ZSTDCompressor(level)

_COMPRESSORS = {}

ZSTDCompressor.register(_COMPRESSORS)
