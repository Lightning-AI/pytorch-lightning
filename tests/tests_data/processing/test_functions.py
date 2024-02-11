import os
import sys
from unittest import mock

import pytest
from lightning.data import walk
from lightning.data.processing.functions import _get_input_dir


@pytest.mark.skipif(sys.platform == "win32", reason="currently not supported for windows.")
def test_get_input_dir(tmpdir, monkeypatch):
    monkeypatch.setattr(os.path, "exists", mock.MagicMock(return_value=True))
    assert _get_input_dir(["/teamspace/studios/here/a", "/teamspace/studios/here/b"]) == "/teamspace/studios/here"

    exists_res = [False, True]

    def fn(*_, **__):
        return exists_res.pop(0)

    monkeypatch.setattr(os.path, "exists", fn)

    with pytest.raises(ValueError, match="The provided item  didn't contain any filepaths."):
        assert _get_input_dir(["", "/teamspace/studios/asd/b"])


def test_walk(tmpdir):
    for i in range(5):
        folder_path = os.path.join(tmpdir, str(i))
        os.makedirs(folder_path, exist_ok=True)
        for j in range(5):
            filepath = os.path.join(folder_path, f"{j}.txt")
            with open(filepath, "w") as f:
                f.write("hello world !")

    walks_os = sorted(os.walk(tmpdir))
    walks_function = sorted(walk(tmpdir))
    assert walks_os == walks_function
