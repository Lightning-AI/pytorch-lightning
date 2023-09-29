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

from abc import ABC, abstractmethod
from io import BytesIO

import numpy as np
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")

if _PIL_AVAILABLE:
    from PIL import Image
    from PIL.JpegImagePlugin import JpegImageFile
else:
    Image = None
    JpegImageFile = None


class Serializer(ABC):
    """The base interface for any serializers.

    A Serializer serialize and deserialize to and from bytes.

    """

    @abstractmethod
    def serialize(self, data: any) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> any:
        pass

    @abstractmethod
    def can_serialize(self, data: any) -> bool:
        pass


class PILSerializer(Serializer):
    """The PILSerializer serialize and deserialize PIL Image to and from bytes."""

    def serialize(self, item: any) -> bytes:
        mode = item.mode.encode("utf-8")
        width, height = item.size
        raw = item.tobytes()
        ints = np.array([width, height, len(mode)], np.uint32)
        return ints.tobytes() + mode + raw

    def deserialize(self, data: bytes) -> any:
        idx = 3 * 4
        width, height, mode_size = np.frombuffer(data[:idx], np.uint32)
        idx2 = idx + mode_size
        mode = data[idx:idx2].decode("utf-8")
        size = width, height
        raw = data[idx2:]
        return Image.frombytes(mode, size, raw)  # pyright: ignore

    def can_serialize(self, item) -> bool:
        pass


class IntSerializer(Serializer):
    """The IntSerializer serialize and deserialize integer to and from bytes."""

    def serialize(self, item: int) -> bytes:
        return str(item).encode("utf-8")

    def deserialize(self, data: bytes) -> int:
        return int(data.decode("utf-8"))

    def can_serialize(self, item) -> bool:
        return isinstance(item, int)


class JPEGSerializer(Serializer):
    """The JPEGSerializer serialize and deserialize JPEG image to and from bytes."""

    def serialize(self, obj: Image) -> bytes:
        if isinstance(obj, JpegImageFile):
            if not hasattr(obj, "filename"):
                raise ValueError(
                    "The JPEG Image's filename isn't defined. HINT: Open the image in your Dataset __getitem__ method."
                )
            with open(obj.filename, "rb") as f:
                return f.read()
        raise TypeError(f"The provided object should be of type {JpegImageFile}. Found {obj}.")

    def deserialize(self, data: bytes) -> Image:
        inp = BytesIO(data)
        return Image.open(inp)

    def can_serialize(self, item) -> bool:
        pass


_SERIALIZERS = {
    "pil": PILSerializer(),
    "int": IntSerializer(),
    "jpeg": JPEGSerializer(),
}
