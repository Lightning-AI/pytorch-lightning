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
import torch
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

    def serialize(self, item: Image) -> bytes:
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
        return isinstance(item, Image.Image) and not isinstance(item, JpegImageFile)


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

    def serialize(self, item: Image) -> bytes:
        if isinstance(item, JpegImageFile):
            if not hasattr(item, "filename"):
                raise ValueError(
                    "The JPEG Image's filename isn't defined. HINT: Open the image in your Dataset __getitem__ method."
                )
            with open(item.filename, "rb") as f:
                return f.read()
        raise TypeError(f"The provided itemect should be of type {JpegImageFile}. Found {item}.")

    def deserialize(self, data: bytes) -> Image:
        inp = BytesIO(data)
        return Image.open(inp)

    def can_serialize(self, item) -> bool:
        return isinstance(item, JpegImageFile)


class BytesSerializer(Serializer):
    """The BytesSerializer serialize and deserialize integer to and from bytes."""

    def serialize(self, item: bytes) -> bytes:
        return item

    def deserialize(self, item: bytes) -> bytes:
        return item

    def can_serialize(self, item: bytes) -> bool:
        return isinstance(item, bytes)


_TORCH_DTYPES_MAPPING = {
    0: torch.float32,
    1: torch.float,
    2: torch.float64,
    3: torch.double,
    4: torch.complex64,
    5: torch.cfloat,
    6: torch.complex128,
    7: torch.cdouble,
    8: torch.float16,
    9: torch.half,
    10: torch.bfloat16,  # Not supported https://github.com/pytorch/pytorch/issues/110285
    11: torch.uint8,
    12: torch.int8,
    13: torch.int16,
    14: torch.short,
    15: torch.int32,
    16: torch.int,
    17: torch.int64,
    18: torch.long,
    19: torch.bool,
}


class TensorSerializer(Serializer):
    """The TensorSerializer serialize and deserialize tensor to and from bytes."""

    def __init__(self):
        super().__init__()
        self._dtype_to_indice = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}

    def serialize(self, item: torch.Tensor) -> bytes:
        dtype_indice = self._dtype_to_indice[item.dtype]
        data = [np.uint32(dtype_indice).tobytes()]
        data.append(np.uint32(len(item.shape)).tobytes())
        for dim in item.shape:
            data.append(np.uint32(dim).tobytes())
        data.append(item.numpy().tobytes())
        return b"".join(data)

    def deserialize(self, data: bytes) -> torch.Tensor:
        dtype_indice = np.frombuffer(data[0:4], np.uint32).item()
        dtype = _TORCH_DTYPES_MAPPING[dtype_indice]
        shape_size = np.frombuffer(data[4:8], np.uint32).item()
        shape = []
        for shape_idx in range(shape_size):
            shape.append(np.frombuffer(data[8 + 4 * shape_idx : 8 + 4 * (shape_idx + 1)], np.uint32).item())
        tensor = torch.frombuffer(data[8 + 4 * (shape_idx + 1) : len(data)], dtype=dtype)
        return torch.reshape(tensor, torch.Size(shape))

    def can_serialize(self, item: torch.Tensor) -> bool:
        return isinstance(item, torch.Tensor)


_SERIALIZERS = {
    "pil": PILSerializer(),
    "int": IntSerializer(),
    "jpeg": JPEGSerializer(),
    "bytes": BytesSerializer(),
    "tensor": TensorSerializer(),
}
