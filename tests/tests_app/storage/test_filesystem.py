import os
import sys

import pytest
from lightning.app.storage import FileSystem


@pytest.mark.skipif(sys.platform == "win32", reason="TODO: Add support for windows")
def test_filesystem(tmpdir):
    fs = FileSystem()

    with open(f"{tmpdir}/a.txt", "w") as f:
        f.write("example")

    os.makedirs(f"{tmpdir}/checkpoints", exist_ok=True)
    with open(f"{tmpdir}/checkpoints/a.txt", "w") as f:
        f.write("example")

    with open(f"{tmpdir}/info.txt", "w") as f:
        f.write("example")

    assert fs.listdir("/") == []
    fs.put(f"{tmpdir}/a.txt", "/a.txt")
    fs.put(f"{tmpdir}/info.txt", "/info.txt")
    assert fs.listdir("/") == ["a.txt"]

    assert fs.isfile("/a.txt")

    fs.put(f"{tmpdir}/checkpoints", "/checkpoints")
    assert not fs.isfile("/checkpoints")
    assert fs.isdir("/checkpoints")
    assert fs.isfile("/checkpoints/a.txt")

    assert fs.listdir("/") == ["a.txt", "checkpoints"]
    assert fs.walk("/") == ["a.txt", "checkpoints/a.txt"]

    os.remove(f"{tmpdir}/a.txt")

    assert not os.path.exists(f"{tmpdir}/a.txt")

    fs.get("/a.txt", f"{tmpdir}/a.txt")

    assert os.path.exists(f"{tmpdir}/a.txt")

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
def test_filesystem_root(tmpdir):
    fs = FileSystem()

    with open(f"{tmpdir}/a.txt", "w") as f:
        f.write("example")

    os.makedirs(f"{tmpdir}/checkpoints", exist_ok=True)
    with open(f"{tmpdir}/checkpoints/a.txt", "w") as f:
        f.write("example")

    assert fs.listdir("/") == []
    fs.put(f"{tmpdir}/a.txt", "/")
    fs.put(f"{tmpdir}/checkpoints", "/")
    assert fs.listdir("/") == ["a.txt", "checkpoints"]
