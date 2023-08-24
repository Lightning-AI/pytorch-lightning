import os
import sys

import pytest

from lightning.app.storage import FileSystem


@pytest.mark.skipif(sys.platform == "win32", reason="TODO: Add support for windows")
def test_filesystem(tmp_path):
    fs = FileSystem()

    with open(f"{tmp_path}/a.txt", "w") as f:
        f.write("example")

    os.makedirs(f"{tmp_path}/checkpoints", exist_ok=True)
    with open(f"{tmp_path}/checkpoints/a.txt", "w") as f:
        f.write("example")

    with open(f"{tmp_path}/info.txt", "w") as f:
        f.write("example")

    assert fs.listdir("/") == []
    fs.put(f"{tmp_path}/a.txt", "/a.txt")
    fs.put(f"{tmp_path}/info.txt", "/info.txt")
    assert fs.listdir("/") == ["a.txt"]

    assert fs.isfile("/a.txt")

    fs.put(f"{tmp_path}/checkpoints", "/checkpoints")
    assert not fs.isfile("/checkpoints")
    assert fs.isdir("/checkpoints")
    assert fs.isfile("/checkpoints/a.txt")

    assert fs.listdir("/") == ["a.txt", "checkpoints"]
    assert fs.walk("/") == ["a.txt", "checkpoints/a.txt"]

    os.remove(f"{tmp_path}/a.txt")

    assert not os.path.exists(f"{tmp_path}/a.txt")

    fs.get("/a.txt", f"{tmp_path}/a.txt")

    assert os.path.exists(f"{tmp_path}/a.txt")

    fs.rm("/a.txt")

    assert fs.listdir("/") == ["checkpoints"]
    fs.rm("/checkpoints/a.txt")
    assert fs.listdir("/") == ["checkpoints"]
    assert fs.walk("/checkpoints") == []
    fs.rm("/checkpoints/")
    assert fs.listdir("/") == []

    with pytest.raises(FileExistsError, match="HERE"):
        fs.put("HERE", "/HERE")

    with pytest.raises(RuntimeError, match="The provided path"):
        fs.listdir("/space")


@pytest.mark.skipif(sys.platform == "win32", reason="TODO: Add support for windows")
def test_filesystem_root(tmp_path):
    fs = FileSystem()

    with open(f"{tmp_path}/a.txt", "w") as f:
        f.write("example")

    os.makedirs(f"{tmp_path}/checkpoints", exist_ok=True)
    with open(f"{tmp_path}/checkpoints/a.txt", "w") as f:
        f.write("example")

    assert fs.listdir("/") == []
    fs.put(f"{tmp_path}/a.txt", "/")
    fs.put(f"{tmp_path}/checkpoints", "/")
    assert fs.listdir("/") == ["a.txt", "checkpoints"]
