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
import sys

import numpy as np
import pytest
from lightning import seed_everything
from lightning.data.streaming.compression import _ZSTD_AVAILABLE
from lightning.data.streaming.reader import BinaryReader
from lightning.data.streaming.sampler import ChunkedIndex
from lightning.data.streaming.writer import BinaryWriter
from lightning.data.utilities.format import _FORMAT_TO_RATIO
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


def test_binary_writer_with_ints_and_chunk_bytes(tmpdir):
    with pytest.raises(FileNotFoundError, match="The provided cache directory `dontexists` doesn't exist."):
        BinaryWriter("dontexists", {})

    match = (
        "The provided compression something_else isn't available"
        if _ZSTD_AVAILABLE
        else "No compresion algorithms are installed."
    )

    with pytest.raises(ValueError, match=match):
        BinaryWriter(tmpdir, {"i": "int"}, compression="something_else")

    binary_writer = BinaryWriter(tmpdir, chunk_bytes=90)

    for i in range(100):
        binary_writer[i] = {"i": i, "i+1": i + 1, "i+2": i + 2}

    assert len(os.listdir(tmpdir)) == 19
    binary_writer.done()
    binary_writer.merge()
    assert len(os.listdir(tmpdir)) == 21

    with open(os.path.join(tmpdir, "index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["chunk_size"] == 6
    assert data["chunks"][1]["chunk_size"] == 5
    assert data["chunks"][-1]["chunk_size"] == 4

    chunk_sizes = np.cumsum([chunk["chunk_size"] for chunk in data["chunks"]])

    reader = BinaryReader(tmpdir, max_cache_size=10 ^ 9)
    for i in range(100):
        for chunk_index, chunk_start in enumerate(chunk_sizes):
            if i >= chunk_start:
                continue
            break
        data = reader.read(ChunkedIndex(i, chunk_index=chunk_index))
        assert data == {"i": i, "i+1": i + 1, "i+2": i + 2}


def test_binary_writer_with_ints_and_chunk_size(tmpdir):
    seed_everything(42)

    with pytest.raises(FileNotFoundError, match="The provided cache directory `dontexists` doesn't exist."):
        BinaryWriter("dontexists", {})

    match = (
        "The provided compression something_else isn't available"
        if _ZSTD_AVAILABLE
        else "No compresion algorithms are installed."
    )

    with pytest.raises(ValueError, match=match):
        BinaryWriter(tmpdir, {"i": "int"}, compression="something_else")

    binary_writer = BinaryWriter(tmpdir, chunk_size=25)

    indices = list(range(100))
    indices = indices[:5] + np.random.permutation(indices[5:]).tolist()

    for i in indices:
        binary_writer[i] = {"i": i, "i+1": i + 1, "i+2": i + 2}

    assert len(os.listdir(tmpdir)) >= 2
    binary_writer.done()
    binary_writer.merge()
    assert len(os.listdir(tmpdir)) == 5

    with open(os.path.join(tmpdir, "index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["chunk_size"] == 25
    assert data["chunks"][1]["chunk_size"] == 25
    assert data["chunks"][-1]["chunk_size"] == 25

    reader = BinaryReader(tmpdir, max_cache_size=10 ^ 9)
    for i in range(100):
        data = reader.read(ChunkedIndex(i, chunk_index=i // 25))
        assert data == {"i": i, "i+1": i + 1, "i+2": i + 2}


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "darwin", reason="Requires: ['pil']")
def test_binary_writer_with_jpeg_and_int(tmpdir):
    """Validate the writer and reader can serialize / deserialize a pair of image and label."""
    from PIL import Image

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    binary_writer = BinaryWriter(cache_dir, chunk_bytes=2 << 12)

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
    binary_writer.done()
    binary_writer.merge()
    assert len(os.listdir(cache_dir)) == 26

    with open(os.path.join(cache_dir, "index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["chunk_size"] == 4
    assert data["chunks"][1]["chunk_size"] == 4
    assert data["chunks"][-1]["chunk_size"] == 4

    reader = BinaryReader(cache_dir, max_cache_size=10 ^ 9)
    for i in range(100):
        data = reader.read(ChunkedIndex(i, chunk_index=i // 4))
        np.testing.assert_array_equal(np.asarray(data["x"]).squeeze(0), imgs[i])
        assert data["y"] == i


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "darwin", reason="Requires: ['pil']")
def test_binary_writer_with_jpeg_filepath_and_int(tmpdir):
    """Validate the writer and reader can serialize / deserialize a pair of image and label."""
    from PIL import Image

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    binary_writer = BinaryWriter(cache_dir, chunk_bytes=2 << 12)

    imgs = []

    for i in range(100):
        path = os.path.join(tmpdir, f"img{i}.jpeg")
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(np_data).convert("L")
        img.save(path, format="jpeg", quality=100)
        img = Image.open(path)
        imgs.append(img)
        binary_writer[i] = {"x": path, "y": i}

    assert len(os.listdir(cache_dir)) == 24
    binary_writer.done()
    binary_writer.merge()
    assert len(os.listdir(cache_dir)) == 26

    with open(os.path.join(cache_dir, "index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["chunk_size"] == 4
    assert data["chunks"][1]["chunk_size"] == 4
    assert data["chunks"][-1]["chunk_size"] == 4
    assert sum([chunk["chunk_size"] for chunk in data["chunks"]]) == 100

    reader = BinaryReader(cache_dir, max_cache_size=10 ^ 9)
    for i in range(100):
        data = reader.read(ChunkedIndex(i, chunk_index=i // 4))
        np.testing.assert_array_equal(np.asarray(data["x"]).squeeze(0), imgs[i])
        assert data["y"] == i


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_binary_writer_with_jpeg_and_png(tmpdir):
    from PIL import Image

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    binary_writer = BinaryWriter(cache_dir, chunk_bytes=2 << 12)

    np_data = np.random.randint(255, size=(28, 28), dtype=np.uint8)
    img = Image.fromarray(np_data).convert("L")
    path = os.path.join(tmpdir, "img.jpeg")
    img.save(path, format="jpeg", quality=100)
    img_jpeg = Image.open(path)

    binary_writer[0] = {"x": img_jpeg, "y": 0}
    binary_writer[1] = {"x": img, "y": 1}

    with pytest.raises(ValueError, match="The data format changed between items"):
        binary_writer[2] = {"x": 2, "y": 1}


def test_writer_human_format(tmpdir):
    for k, v in _FORMAT_TO_RATIO.items():
        binary_writer = BinaryWriter(tmpdir, chunk_bytes=f"{1}{k}")
        assert binary_writer._chunk_bytes == v

    binary_writer = BinaryWriter(tmpdir, chunk_bytes="64MB")
    assert binary_writer._chunk_bytes == 64000000
