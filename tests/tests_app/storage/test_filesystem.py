import os

import pytest

from lightning.app.storage import FileSystem


def test_filesystem(tmpdir):
    fs = FileSystem()

    with open(f"{tmpdir}/a.txt", "w") as f:
        f.write("example")

    os.makedirs(f"{tmpdir}/checkpoints", exist_ok=True)
    with open(f"{tmpdir}/checkpoints/a.txt", "w") as f:
        f.write("example")

    with open(f"{tmpdir}/info.txt", "w") as f:
        f.write("example")

    assert fs.list(".") == []
    fs.put(f"{tmpdir}/a.txt", "a.txt")
    fs.put(f"{tmpdir}/info.txt", "info.txt")
    assert fs.list(".") == ["a.txt"]

    fs.put(f"{tmpdir}/checkpoints", "checkpoints")
    assert fs.list(".") == ["a.txt", "checkpoints/a.txt"]

    fs.delete("a.txt")

    assert fs.list(".") == ["checkpoints/a.txt"]
    fs.delete("checkpoints/a.txt")
    assert fs.list(".") == []

    with pytest.raises(FileExistsError, match="HERE"):
        fs.put("HERE", "HERE")

    with pytest.raises(FileNotFoundError, match="The provided path"):
        fs.list("space")