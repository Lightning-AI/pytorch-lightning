import os
import pathlib
from unittest import mock
from unittest.mock import Mock

import lightning.app
import pytest
from lightning.app.storage.copier import _Copier, _copy_files
from lightning.app.storage.path import Path
from lightning.app.storage.requests import _ExistsRequest, _GetRequest
from lightning.app.testing.helpers import _MockQueue


class MockPatch:
    @staticmethod
    def _handle_get_request(work, request):
        return Path._handle_get_request(work, request)

    @staticmethod
    def _handle_exists_request(work, request):
        return Path._handle_exists_request(work, request)


@mock.patch("lightning.app.storage.path.pathlib.Path.is_dir")
@mock.patch("lightning.app.storage.path.pathlib.Path.stat")
@mock.patch("lightning.app.storage.copier._filesystem")
def test_copier_copies_all_files(fs_mock, stat_mock, dir_mock, tmpdir):
    """Test that the Copier calls the copy with the information provided in the request."""
    stat_mock().st_size = 0
    dir_mock.return_value = False
    copy_request_queue = _MockQueue()
    copy_response_queue = _MockQueue()
    work = mock.Mock()
    work.name = MockPatch()
    work._paths = {"file": {"source": "src", "path": "file", "hash": "123", "destination": "dest", "name": "name"}}
    with mock.patch.dict(os.environ, {"SHARED_MOUNT_DIRECTORY": str(tmpdir / ".shared")}):
        copier = _Copier(work, copy_request_queue=copy_request_queue, copy_response_queue=copy_response_queue)
        request = _GetRequest(source="src", path="file", hash="123", destination="dest", name="name")
        copy_request_queue.put(request)
        copier.run_once()
        fs_mock().put.assert_called_once_with("file", tmpdir / ".shared" / "123")


@mock.patch("lightning.app.storage.path.pathlib.Path.is_dir")
@mock.patch("lightning.app.storage.path.pathlib.Path.stat")
def test_copier_handles_exception(stat_mock, dir_mock, monkeypatch):
    """Test that the Copier captures exceptions from the file copy and forwards them through the queue without raising
    it."""
    stat_mock().st_size = 0
    dir_mock.return_value = False
    copy_request_queue = _MockQueue()
    copy_response_queue = _MockQueue()
    fs = mock.Mock()
    fs.exists.return_value = False
    fs.put = mock.Mock(side_effect=OSError("Something went wrong"))
    monkeypatch.setattr(lightning.app.storage.copier, "_filesystem", mock.Mock(return_value=fs))

    work = mock.Mock()
    work.name = MockPatch()
    work._paths = {"file": {"source": "src", "path": "file", "hash": "123", "destination": "dest", "name": "name"}}
    copier = _Copier(work, copy_request_queue=copy_request_queue, copy_response_queue=copy_response_queue)
    request = _GetRequest(source="src", path="file", hash="123", destination="dest", name="name")
    copy_request_queue.put(request)
    copier.run_once()
    response = copy_response_queue.get()
    assert type(response.exception) is OSError
    assert response.exception.args[0] == "Something went wrong"


def test_copier_existence_check(tmpdir):
    """Test that the Copier responds to an existence check request."""
    copy_request_queue = _MockQueue()
    copy_response_queue = _MockQueue()

    work = mock.Mock()
    work.name = MockPatch()
    work._paths = {
        "file": {
            "source": "src",
            "path": str(tmpdir / "notexists"),
            "hash": "123",
            "destination": "dest",
            "name": "name",
        }
    }

    copier = _Copier(work, copy_request_queue=copy_request_queue, copy_response_queue=copy_response_queue)

    # A Path that does NOT exist
    request = _ExistsRequest(source="src", path=str(tmpdir / "notexists"), destination="dest", name="name", hash="123")
    copy_request_queue.put(request)
    copier.run_once()
    response = copy_response_queue.get()
    assert response.exists is False

    # A Path that DOES exist
    request = _ExistsRequest(source="src", path=str(tmpdir), destination="dest", name="name", hash="123")
    copy_request_queue.put(request)
    copier.run_once()
    response = copy_response_queue.get()
    assert response.exists is True


def test_copy_files(tmpdir):
    """Test that the `test_copy_files` utility can handle both files and folders when the destination does not
    exist."""
    # copy from a src that does not exist
    src = pathlib.Path(tmpdir, "dir1")
    dst = pathlib.Path(tmpdir, "dir2")
    with pytest.raises(FileNotFoundError):
        _copy_files(src, dst)

    # copy to a dst dir that does not exist
    src.mkdir()
    (src / "empty.txt").touch()
    assert not dst.exists()
    _copy_files(src, dst)
    assert dst.is_dir()

    # copy to a destination dir that already exists (no error should be raised)
    _copy_files(src, dst)
    assert dst.is_dir()

    # copy file to a dst that does not exist
    src = pathlib.Path(tmpdir, "dir3", "src-file.txt")
    dst = pathlib.Path(tmpdir, "dir4", "dst-file.txt")
    src.parent.mkdir(parents=True)
    src.touch()
    assert not dst.exists()
    _copy_files(src, dst)
    assert dst.is_file()


def test_copy_files_with_exception(tmpdir):
    """Test that the `test_copy_files` utility properly raises exceptions from within the ThreadPoolExecutor."""
    fs_mock = Mock()
    fs_mock().put = Mock(side_effect=ValueError("error from thread"))

    src = pathlib.Path(tmpdir, "src")
    src.mkdir()
    assert src.is_dir()
    pathlib.Path(src, "file.txt").touch()
    dst = pathlib.Path(tmpdir, "dest")

    with mock.patch("lightning.app.storage.copier._filesystem", fs_mock), pytest.raises(
        ValueError, match="error from thread"
    ):
        _copy_files(src, dst)
