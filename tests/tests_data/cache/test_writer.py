import json
import os

import pytest

from lightning.data.cache.reader import BinaryReader
from lightning.data.cache.writer import BinaryWriter


def test_binary_writer(tmpdir):
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
        data = reader.read(i, 0)
        assert data == {"i": i, "i+1": i + 1, "i+2": i + 2}
