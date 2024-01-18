from unittest import mock

import pytest
from lightning.data.streaming.functions import _get_input_dir, os


def test_get_input_dir(tmpdir, monkeypatch):
    monkeypatch.setattr(os.path, "exists", mock.MagicMock(return_value=True))
    assert _get_input_dir(["/teamspace/studios/here/a", "/teamspace/studios/here/b"]) == "/teamspace/studios/here"

    exists_res = [False, True]

    def fn(*_, **__):
        return exists_res.pop(0)

    monkeypatch.setattr(os.path, "exists", fn)

    with pytest.raises(ValueError, match="The provided item  didn't contain any filepaths."):
        assert _get_input_dir(["", "/teamspace/studios/asd/b"])
