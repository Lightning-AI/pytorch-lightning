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
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from io import BytesIO
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")
_TORCH_VISION_AVAILABLE = RequirementCache("torchvision")

if _PIL_AVAILABLE:
    from PIL import Image
    from PIL.JpegImagePlugin import JpegImageFile
else:
    Image = None
    JpegImageFile = None

if _TORCH_VISION_AVAILABLE:
    from torchvision.io import decode_jpeg


class Serializer(ABC):
    """The base interface for any serializers.

    A Serializer serialize and deserialize to and from bytes.

    """

    @abstractmethod
    def serialize(self, data: Any) -> Tuple[bytes, Optional[str]]:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        pass

    @abstractmethod
    def can_serialize(self, data: Any) -> bool:
        pass


class PILSerializer(Serializer):
    """The PILSerializer serialize and deserialize PIL Image to and from bytes."""

    def serialize(self, item: Image) -> Tuple[bytes, Optional[str]]:
        mode = item.mode.encode("utf-8")
        width, height = item.size
        raw = item.tobytes()
        ints = np.array([width, height, len(mode)], np.uint32)
        return ints.tobytes() + mode + raw, None

    def deserialize(self, data: bytes) -> Any:
        idx = 3 * 4
        width, height, mode_size = np.frombuffer(data[:idx], np.uint32)
        idx2 = idx + mode_size
        mode = data[idx:idx2].decode("utf-8")
        size = width, height
        raw = data[idx2:]
        return Image.frombytes(mode, size, raw)  # pyright: ignore

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, Image.Image) and not isinstance(item, JpegImageFile)


class IntSerializer(Serializer):
    """The IntSerializer serialize and deserialize integer to and from bytes."""

    def serialize(self, item: int) -> Tuple[bytes, Optional[str]]:
        return str(item).encode("utf-8"), None

    def deserialize(self, data: bytes) -> int:
        return int(data.decode("utf-8"))

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, int)


class JPEGSerializer(Serializer):
    """The JPEGSerializer serialize and deserialize JPEG image to and from bytes."""

    def serialize(self, item: Image) -> Tuple[bytes, Optional[str]]:
        if isinstance(item, JpegImageFile):
            if not hasattr(item, "filename"):
                raise ValueError(
                    "The JPEG Image's filename isn't defined. HINT: Open the image in your Dataset __getitem__ method."
                )
            with open(item.filename, "rb") as f:
                return f.read(), None
        raise TypeError(f"The provided itemect should be of type {JpegImageFile}. Found {item}.")

    def deserialize(self, data: bytes) -> Union[JpegImageFile, torch.Tensor]:
        if _TORCH_VISION_AVAILABLE:
            array = torch.frombuffer(data, dtype=torch.uint8)
            return decode_jpeg(array)

        inp = BytesIO(data)
        return Image.open(inp)

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, JpegImageFile)


class BytesSerializer(Serializer):
    """The BytesSerializer serialize and deserialize integer to and from bytes."""

    def serialize(self, item: bytes) -> Tuple[bytes, Optional[str]]:
        return item, None

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

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indice = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}

    def serialize(self, item: torch.Tensor) -> Tuple[bytes, Optional[str]]:
        dtype_indice = self._dtype_to_indice[item.dtype]
        data = [np.uint32(dtype_indice).tobytes()]
        data.append(np.uint32(len(item.shape)).tobytes())
        for dim in item.shape:
            data.append(np.uint32(dim).tobytes())
        data.append(item.numpy().tobytes())
        return b"".join(data), None

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
        return isinstance(item, torch.Tensor) and type(item) == torch.Tensor


class PickleSerializer(Serializer):
    """The PickleSerializer serialize and deserialize python objects to and from bytes."""

    def serialize(self, item: Any) -> Tuple[bytes, Optional[str]]:
        return pickle.dumps(item), None

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    def can_serialize(self, _: Any) -> bool:
        return True


class FileSerializer(Serializer):
    def serialize(self, filepath: str) -> Tuple[bytes, Optional[str]]:
        _, file_extension = os.path.splitext(filepath)
        with open(filepath, "rb") as f:
            return f.read(), file_extension.replace(".", "").lower()

    def deserialize(self, data: bytes) -> Any:
        pass

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, str) and os.path.exists(data)


_SERIALIZERS = OrderedDict(
    **{
        "file": FileSerializer(),
        "pil": PILSerializer(),
        "int": IntSerializer(),
        "jpeg": JPEGSerializer(),
        "bytes": BytesSerializer(),
        "tensor": TensorSerializer(),
        "pickle": PickleSerializer(),
    }
)
