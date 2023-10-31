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
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from lightning_utilities.core.imports import RequirementCache

from lightning.data.streaming.constants import _TORCH_DTYPES_MAPPING

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
    from torchvision.transforms.functional import pil_to_tensor


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

    def setup(self, metadata: Any) -> None:
        pass


class PILSerializer(Serializer):
    """The PILSerializer serialize and deserialize PIL Image to and from bytes."""

    def serialize(self, item: Image) -> Tuple[bytes, Optional[str]]:
        mode = item.mode.encode("utf-8")
        width, height = item.size
        raw = item.tobytes()
        ints = np.array([width, height, len(mode)], np.uint32)
        return ints.tobytes() + mode + raw, None

    @classmethod
    def deserialize(cls, data: bytes) -> Any:
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
            try:
                return decode_jpeg(array)
            except RuntimeError:
                # Note: Some datasets like Imagenet contains some PNG images with JPEG extension, so we fallback to PIL
                pass

        img = PILSerializer.deserialize(data)
        if _TORCH_VISION_AVAILABLE:
            img = pil_to_tensor(img)
        return img

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
        data.append(item.numpy().tobytes(order="C"))
        return b"".join(data), None

    def deserialize(self, data: bytes) -> torch.Tensor:
        dtype_indice = np.frombuffer(data[0:4], np.uint32).item()
        dtype = _TORCH_DTYPES_MAPPING[dtype_indice]
        shape_size = np.frombuffer(data[4:8], np.uint32).item()
        shape = []
        for shape_idx in range(shape_size):
            shape.append(np.frombuffer(data[8 + 4 * shape_idx : 8 + 4 * (shape_idx + 1)], np.uint32).item())
        tensor = torch.frombuffer(data[8 + 4 * (shape_idx + 1) : len(data)], dtype=dtype)
        shape = torch.Size(shape)
        if tensor.shape == shape:
            return tensor
        return torch.reshape(tensor, shape)

    def can_serialize(self, item: torch.Tensor) -> bool:
        return isinstance(item, torch.Tensor) and type(item) == torch.Tensor and len(item.shape) > 1


class NoHeaderTensorSerializer(Serializer):
    """The TensorSerializer serialize and deserialize tensor to and from bytes."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indice = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}
        self._dtype: Optional[torch.dtype] = None

    def setup(self, data_format: str) -> None:
        self._dtype = _TORCH_DTYPES_MAPPING[int(data_format.split(":")[1])]

    def serialize(self, item: torch.Tensor) -> Tuple[bytes, Optional[str]]:
        dtype_indice = self._dtype_to_indice[item.dtype]
        return item.numpy().tobytes(order="C"), f"no_header_tensor:{dtype_indice}"

    def deserialize(self, data: bytes) -> torch.Tensor:
        assert self._dtype
        return torch.frombuffer(data, dtype=self._dtype)

    def can_serialize(self, item: torch.Tensor) -> bool:
        return isinstance(item, torch.Tensor) and type(item) == torch.Tensor and len(item.shape) == 1


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
        "no_header_tensor": NoHeaderTensorSerializer(),
        "tensor": TensorSerializer(),
        "pickle": PickleSerializer(),
    }
)
