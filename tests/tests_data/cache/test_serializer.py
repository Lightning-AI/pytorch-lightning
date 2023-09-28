import os

import numpy as np
import pytest
from lightning.data.cache.serializers import _SERIALIZERS, IntSerializer, JPEGSerializer, PILSerializer
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


def test_serializers():
    assert sorted(_SERIALIZERS) == ["int", "jpeg", "pil"]


def test_int_serializer():
    serializer = IntSerializer()

    for i in range(100):
        data = serializer.serialize(i)
        assert isinstance(data, bytes)
        assert i == serializer.deserialize(data)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_jpeg_serializer(mode, tmpdir):
    serializer = JPEGSerializer()

    from PIL import Image

    path = os.path.join(tmpdir, "img.jpeg")

    size = {"RGB": (28, 28, 3), "L": (28, 28)}[mode]
    np_data = np.random.randint(255, size=size, dtype=np.uint8)
    img = Image.fromarray(np_data).convert(mode)

    np.testing.assert_array_equal(np_data, np.array(img))

    with pytest.raises(TypeError, match="PIL.JpegImagePlugin.JpegImageFile"):
        serializer.serialize(img)

    # from the JPEG image directly
    img.save(path, format="jpeg", quality=100)
    img = Image.open(path)

    data = serializer.serialize(img)
    assert isinstance(data, bytes)
    deserialized_img = np.asarray(serializer.deserialize(data))
    assert np.array_equal(np.asarray(img), np.array(deserialized_img))

    # read bytes from the file
    with open(path, "rb") as f:
        data = f.read()

    assert isinstance(data, bytes)
    deserialized_img = np.asarray(serializer.deserialize(data))

    assert np.array_equal(np.asarray(img), np.array(deserialized_img))


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
@pytest.mark.parametrize("mode", ["I", "L", "RGB"])
def test_pil_serializer(mode):
    serializer = PILSerializer()

    from PIL import Image

    np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
    img = Image.fromarray(np_data).convert(mode)

    data = serializer.serialize(img)
    assert isinstance(data, bytes)

    deserialized_img = serializer.deserialize(data)
    deserialized_img = deserialized_img.convert("I")
    np_dec_data = np.asarray(deserialized_img, dtype=np.uint32)
    assert isinstance(deserialized_img, Image.Image)

    # Validate data content
    assert np.array_equal(np_data, np_dec_data)
