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
from lightning.data.cache.reader import BinaryReader
from lightning.data.cache.writer import BinaryWriter, get_cloud_path
from lightning_utilities.core.imports import RequirementCache

_PIL_AVAILABLE = RequirementCache("PIL")


def test_binary_writer_with_ints(tmpdir):
    with pytest.raises(FileNotFoundError, match="The provided cache directory `dontexists` doesn't exist."):
        BinaryWriter("dontexists", {})

    with pytest.raises(ValueError, match="No compresion algorithms are installed."):
        BinaryWriter(tmpdir, {"i": "int"}, compression="something_else")

    binary_writer = BinaryWriter(tmpdir, chunk_size=90)

    for i in range(100):
        binary_writer[i] = {"i": i, "i+1": i + 1, "i+2": i + 2}

    assert len(os.listdir(tmpdir)) == 19
    binary_writer.done()
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
    binary_writer = BinaryWriter(cache_dir, chunk_size=2 << 12)

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


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_binary_writer_config(monkeypatch):
    assert get_cloud_path("") is None

    monkeypatch.setenv("LIGHTNING_CLUSTER_ID", "cluster_id")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")
    monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "cloud_space_id")

    prefix = "s3://cluster_id/projects/project_id/cloudspaces/cloud_space_id/code/content/"

    assert get_cloud_path("") == prefix
    assert get_cloud_path("~") == prefix
    assert get_cloud_path("~/") == prefix
    assert get_cloud_path("/") == prefix
    assert get_cloud_path("/data") == f"{prefix}data"
    assert get_cloud_path("~/data") == f"{prefix}data"
    assert get_cloud_path("/teamspace/studios/this_studio/data") == f"{prefix}data"
