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

import io
import os
import sys
from time import time

import numpy as np
import pytest
import torch
from lightning import seed_everything
from lightning.data.streaming.serializers import (
    _AV_AVAILABLE,
    _NUMPY_DTYPES_MAPPING,
    _SERIALIZERS,
    _TORCH_DTYPES_MAPPING,
    _TORCH_VISION_AVAILABLE,
    IntegerSerializer,
    JPEGSerializer,
    NoHeaderNumpySerializer,
    NoHeaderTensorSerializer,
    NumpySerializer,
    PickleSerializer,
    PILSerializer,
    TensorSerializer,
    VideoSerializer,
)
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


def test_serializers():
    keys = list(_SERIALIZERS.keys())
    assert keys == [
        'str', 'int', 'float', 'video', 'tif', 'file', 'pil', 'jpeg',
        'bytes', 'no_header_numpy', 'numpy', 'no_header_tensor', 'tensor', 'pickle']


def test_int_serializer():
    serializer = IntegerSerializer()

    for i in range(100):
        data, _ = serializer.serialize(i)
        assert isinstance(data, bytes)
        assert i == serializer.deserialize(data)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
@pytest.mark.parametrize("mode", ["I", "L", "RGB"])
def test_pil_serializer(mode):
    serializer = PILSerializer()

    from PIL import Image

    np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
    img = Image.fromarray(np_data).convert(mode)

    data, _ = serializer.serialize(img)
    assert isinstance(data, bytes)

    deserialized_img = serializer.deserialize(data)
    deserialized_img = deserialized_img.convert("I")
    np_dec_data = np.asarray(deserialized_img, dtype=np.uint32)
    assert isinstance(deserialized_img, Image.Image)

    # Validate data content
    assert np.array_equal(np_data, np_dec_data)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_jpeg_serializer():
    serializer = JPEGSerializer()

    from PIL import Image

    array = np.random.randint(255, size=(28, 28, 3), dtype=np.uint8)
    img = Image.fromarray(array)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    img = Image.open(io.BytesIO(img_bytes))

    data, _ = serializer.serialize(img)
    assert isinstance(data, bytes)

    deserialized_img = serializer.deserialize(data)
    assert deserialized_img.shape == torch.Size([3, 28, 28])


@pytest.mark.flaky(reruns=3)
@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_tensor_serializer():
    seed_everything(42)

    serializer_tensor = TensorSerializer()
    serializer_pickle = PickleSerializer()

    ratio_times = []
    ratio_bytes = []
    shapes = [(10,), (10, 10), (10, 10, 10), (10, 10, 10, 5), (10, 10, 10, 5, 4)]
    for dtype in _TORCH_DTYPES_MAPPING.values():
        for shape in shapes:
            # Not serializable for some reasons
            if dtype in [torch.bfloat16]:
                continue
            tensor = torch.ones(shape, dtype=dtype)

            t0 = time()
            data, _ = serializer_tensor.serialize(tensor)
            deserialized_tensor = serializer_tensor.deserialize(data)
            tensor_time = time() - t0
            tensor_bytes = len(data)

            assert deserialized_tensor.dtype == dtype
            assert torch.equal(tensor, deserialized_tensor)

            t1 = time()
            data, _ = serializer_pickle.serialize(tensor)
            deserialized_tensor = serializer_pickle.deserialize(data)
            pickle_time = time() - t1
            pickle_bytes = len(data)

            assert deserialized_tensor.dtype == dtype
            assert torch.equal(tensor, deserialized_tensor)

            ratio_times.append(pickle_time / tensor_time)
            ratio_bytes.append(pickle_bytes / tensor_bytes)

    assert np.mean(ratio_times) > 1.6
    assert np.mean(ratio_bytes) > 2


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_numpy_serializer():
    seed_everything(42)

    serializer_tensor = NumpySerializer()

    shapes = [(10,), (10, 10), (10, 10, 10), (10, 10, 10, 5), (10, 10, 10, 5, 4)]
    for dtype in _NUMPY_DTYPES_MAPPING.values():
        # Those types aren't supported
        if dtype.name in ["object", "bytes", "str", "void"]:
            continue
        for shape in shapes:
            tensor = np.ones(shape, dtype=dtype)
            data, _ = serializer_tensor.serialize(tensor)
            deserialized_tensor = serializer_tensor.deserialize(data)
            assert deserialized_tensor.dtype == dtype
            np.testing.assert_equal(tensor, deserialized_tensor)


def test_assert_bfloat16_tensor_serializer():
    serializer = TensorSerializer()
    tensor = torch.ones((10,), dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="Got unsupported ScalarType BFloat16"):
        serializer.serialize(tensor)


def test_assert_no_header_tensor_serializer():
    serializer = NoHeaderTensorSerializer()
    t = torch.ones((10,))
    data, name = serializer.serialize(t)
    assert name == "no_header_tensor:1"
    assert serializer._dtype is None
    serializer.setup(name)
    assert serializer._dtype == torch.float32
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)


def test_assert_no_header_numpy_serializer():
    serializer = NoHeaderNumpySerializer()
    t = np.ones((10,))
    assert serializer.can_serialize(t)
    data, name = serializer.serialize(t)
    assert name == "no_header_numpy:10"
    assert serializer._dtype is None
    serializer.setup(name)
    assert serializer._dtype == np.dtype("float64")
    new_t = serializer.deserialize(data)
    np.testing.assert_equal(t, new_t)


@pytest.mark.skipif(
    condition=not _TORCH_VISION_AVAILABLE or not _AV_AVAILABLE, reason="Requires: ['torchvision', 'av']"
)
def test_wav_deserialization(tmpdir):
    from torch.hub import download_url_to_file

    video_file = os.path.join(tmpdir, "video.wav")
    key = "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa E501
    download_url_to_file(f"https://download.pytorch.org/torchaudio/{key}", video_file)

    serializer = VideoSerializer()
    assert serializer.can_serialize(video_file)
    data, name = serializer.serialize(video_file)
    assert len(data) / 1024 / 1024 == 0.10380172729492188
    assert name == "wav"
    vframes, aframes, info = serializer.deserialize(data)
    assert vframes.shape == torch.Size([0, 1, 1, 3])
    assert aframes.shape == torch.Size([1, 54400])
    assert info == {"audio_fps": 16000}
