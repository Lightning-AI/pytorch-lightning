import os

import pytest

from lightning_app.storage import FileSystem


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

    fs.put(f"{tmpdir}/checkpoints", "/checkpoints")
    assert fs.listdir("/") == ["a.txt", "checkpoints"]
    assert fs.walk("/") == ["a.txt", "checkpoints/a.txt"]

    os.remove(f"{tmpdir}/a.txt")

    assert not os.path.exists(f"{tmpdir}/a.txt")

    fs.get(f"/a.txt", f"{tmpdir}/a.txt")

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
