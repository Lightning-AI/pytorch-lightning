import json
import os

import numpy as np
import pytest
from lightning.data.cache.reader import BinaryReader
from lightning.data.cache.writer import BinaryWriter
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


def test_binary_writer_with_ints(tmpdir):
    with pytest.raises(FileNotFoundError, match="The provided cache directory `dontexists` doesn't exist."):
        BinaryWriter("dontexists", {})

    with pytest.raises(ValueError, match="The provided data format shouldn't be empty."):
        BinaryWriter(tmpdir, {})

    with pytest.raises(ValueError, match="['int', 'jpeg', 'pil']"):
        BinaryWriter(tmpdir, {"i": "random"})

    with pytest.raises(ValueError, match="No compresion algorithms are installed."):
        BinaryWriter(tmpdir, {"i": "int"}, compression="something_else")

    binary_writer = BinaryWriter(tmpdir, {"i": "int", "i+1": "int", "i+2": "int"}, chunk_size=90)

    for i in range(100):
        binary_writer[i] = {"i": i, "i+1": i + 1, "i+2": i + 2}

    assert len(os.listdir(tmpdir)) == 19
    binary_writer.done(0)
    assert len(os.listdir(tmpdir)) == 21

    with open(os.path.join(tmpdir, "0.index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["samples"] == 6
    assert data["chunks"][1]["samples"] == 5
    assert data["chunks"][-1]["samples"] == 4

    reader = BinaryReader(tmpdir)
    for i in range(100):
        data = reader.read(i)
        assert data == {"i": i, "i+1": i + 1, "i+2": i + 2}


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_binary_writer_with_jpeg_and_int(tmpdir):
    """Validate the writer and reader can serialize / deserialize a pair of image and label."""
    from PIL import Image

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    binary_writer = BinaryWriter(cache_dir, {"x": "jpeg", "y": "int"}, chunk_size=2 << 12)

    imgs = []

    for i in range(100):
        path = os.path.join(tmpdir, f"img{i}.jpeg")
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(np_data).convert("L")
        img.save(path, format="jpeg", quality=100)
        img = Image.open(path)
        imgs.append(img)
        binary_writer[i] = {"x": img, "y": i}

    assert len(os.listdir(cache_dir)) == 24
    binary_writer.done(0)
    assert len(os.listdir(cache_dir)) == 26

    with open(os.path.join(cache_dir, "0.index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["samples"] == 4
    assert data["chunks"][1]["samples"] == 4
    assert data["chunks"][-1]["samples"] == 4

    reader = BinaryReader(cache_dir)
    for i in range(100):
        data = reader.read(i)
        assert data["x"] == imgs[i]
        assert data["y"] == i
