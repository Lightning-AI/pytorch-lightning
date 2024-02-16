import json
import os
import pathlib
import pickle
import sys
from re import escape
from time import sleep
from unittest import TestCase, mock
from unittest.mock import MagicMock, Mock

import pytest
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.runners import MultiProcessRuntime
from lightning.app.storage.path import (
    Path,
    _artifacts_path,
    _filesystem,
    _is_lit_path,
    _shared_storage_path,
    _storage_root_dir,
)
from lightning.app.storage.requests import _ExistsResponse, _GetResponse
from lightning.app.testing.helpers import EmptyWork, _MockQueue, _RunIf
from lightning.app.utilities.app_helpers import LightningJSONEncoder
from lightning.app.utilities.component import _context
from lightning.app.utilities.imports import _IS_WINDOWS, _is_s3fs_available


def test_path_instantiation():
    assert Path() == pathlib.Path()
    assert Path("a/b") == pathlib.Path("a/b")
    assert Path("a", "b") == pathlib.Path("a", "b")
    assert Path(pathlib.Path("a"), pathlib.Path("b")) == pathlib.Path("a/b")
    assert Path(Path(Path("a/b"))) == pathlib.Path("a/b")

    path = Path()
    assert path._origin is path._consumer is path._request_queue is path._response_queue is None

    folder = Path("x/y/z")
    folder._origin = "origin"
    folder._consumer = "consumer"

    # from parts where the first is a Lightning Path and the other(s) are string
    file = Path(folder, "file.txt")
    assert file._origin == "origin"
    assert file._consumer == "consumer"

    # from parts that are instance of Path and have no origin
    file = Path(folder, Path("file.txt"))
    assert file._origin == "origin"
    assert file._consumer == "consumer"

    # from parts that are instance of Path and have a different origin than the top folder
    filename = Path("file.txt")
    filename._origin = "different"
    with pytest.raises(TypeError, match="Tried to instantiate a Lightning Path from multiple other Paths"):
        Path(folder, filename)

    # from parts that are instance of Path and have the SAME origin as the top folder
    filename = Path("file.txt")
    filename._origin = "origin"
    file = Path(folder, filename)
    assert file._origin == "origin"
    assert file._consumer == "consumer"


def test_path_instantiation_lit():
    assert Path("lit://") == _storage_root_dir()
    assert Path("lit://a/b") == pathlib.Path(_storage_root_dir(), "a/b")
    assert Path("lit://", "a", "b") == pathlib.Path(_storage_root_dir(), "a", "b")
    assert Path("lit://", pathlib.Path("a"), pathlib.Path("b")) == pathlib.Path(_storage_root_dir(), "a/b")
    assert Path(Path(Path("lit://a/b"))) == pathlib.Path(_storage_root_dir(), "a", "b")
    assert str(Path("lit://lit-path")) == os.path.join(_storage_root_dir(), "lit-path")


def test_is_lit_path():
    assert not _is_lit_path("lit")
    assert not _is_lit_path(Path("lit"))
    assert _is_lit_path("lit://")
    assert _is_lit_path(Path("lit://"))
    assert _is_lit_path("lit://a/b/c")
    assert _is_lit_path(Path("lit://a/b/c"))
    assert _is_lit_path(_storage_root_dir())


def test_path_copy():
    """Test that Path creates an exact copy when passing a Path instance to the constructor."""
    path = Path("x/y/z")
    path._origin = "origin"
    path._consumer = "consumer"
    path._request_queue = Mock()
    path._response_queue = Mock()
    path_copy = Path(path)
    assert path_copy._origin == path._origin
    assert path_copy._consumer == path._consumer
    assert path_copy._request_queue == path._request_queue
    assert path_copy._response_queue == path._response_queue


def test_path_inheritance():
    """Test that the Lightning Path is a drop-in replacement for pathlib.Path without compromises."""
    file = Path("file.txt")
    pathlibfile = pathlib.Path("file.txt")
    assert file == pathlibfile
    assert isinstance(file, Path)
    assert isinstance(file, pathlib.Path)

    folder = Path("./x/y")
    file = folder / "file.txt"
    assert isinstance(file, Path)

    file.with_suffix(".png")
    assert isinstance(file, Path)


def test_path_concatenation():
    """Test that path concatentaions keep the properties of the paths on the right-hand side of the join."""
    folder = Path("x/y/z")
    folder._origin = "origin"
    folder._consumer = "consumer"
    other = Path("other")

    # test __truediv__ when Path is on the left-hand side
    file = folder / other / "more" / "file.txt"
    assert file._origin == "origin"
    assert file._consumer == "consumer"

    # test __rtruediv__ when Path is on the right-hand side
    switched = pathlib.Path("/") / folder
    assert isinstance(switched, Path)
    assert file._origin == "origin"
    assert file._consumer == "consumer"


def test_path_with_replacement():
    """Test that the ``Path.with_*`` modifiers keep the properties."""
    folder = Path("x", "y", "z")
    folder._origin = "origin"
    folder._consumer = "consumer"

    # with_name
    file = folder.with_name("file.txt")
    assert str(file) == os.path.join("x", "y", "file.txt")
    assert file._origin == "origin"
    assert file._consumer == "consumer"

    # with_suffix
    file = file.with_suffix(".png")
    assert str(file) == os.path.join("x", "y", "file.png")
    assert file._origin == "origin"
    assert file._consumer == "consumer"

    # relative_to
    rel_path = folder.relative_to("x")
    assert str(rel_path) == os.path.join("y", "z")
    assert rel_path._origin == "origin"
    assert rel_path._consumer == "consumer"


@_RunIf(min_python="3.9")
def test_path_with_stem_replacement():
    """Test that the ``Path.with_stem`` modifier keep the properties.

    This is only available in Python 3.9+.

    """
    file = Path("x", "y", "file.txt")
    file._origin = "origin"
    file._consumer = "consumer"
    file = file.with_stem("text")
    assert str(file) == os.path.join("x", "y", "text.txt")
    assert file._origin == "origin"
    assert file._consumer == "consumer"


def test_path_parents():
    """Test that the ``Path.parent`` and ``Path.parent`` properties return Paths that inherit the origin and consumer
    attributes."""
    path = Path("a", "b", "c", "d")
    path._origin = "origin"
    path._consumer = "consumer"

    # .parent
    assert isinstance(path.parent, Path)
    assert str(path.parent) == os.path.join("a", "b", "c")
    assert path.parent._origin == "origin"
    assert path.parent._consumer == "consumer"

    # .parents
    assert path.parents == [Path("a", "b", "c"), Path("a", "b"), Path("a"), Path(".")]
    assert all(parent._origin == "origin" for parent in path.parents)
    assert all(parent._consumer == "consumer" for parent in path.parents)


def test_path_hash():
    """Test that the value of the Path hash is a function of the path name and the origin."""
    # a path without origin has no hash
    assert Path("one").hash is Path("two").hash is None

    # identical paths with identical origins have the same hash
    path1 = Path("one")
    path2 = Path("one")
    path1._origin = "origin1"
    path1._consumer = "consumer1"
    path2._origin = "origin1"
    path1._consumer = "consumer2"
    assert path1.hash == path2.hash

    # identical paths with different origins have different hash
    path2._origin = "origin2"
    assert path1.hash != path2.hash

    # different paths but same owner yields a different hash
    path1 = Path("one")
    path2 = Path("other")
    path1._origin = "same"
    path2._origin = "same"
    assert path1.hash != path2.hash


def test_path_pickleable():
    path = Path("a/b/c.txt")
    path._origin = "root.x.y.z"
    path._consumer = "root.p.q.r"
    path._request_queue = Mock()
    path._response_queue = Mock()
    loaded = pickle.loads(pickle.dumps(path))
    assert isinstance(loaded, Path)
    assert loaded == path
    assert loaded._origin == path._origin
    assert loaded._consumer == path._consumer
    assert loaded._request_queue is None
    assert loaded._response_queue is None


def test_path_json_serializable():
    path = Path("a/b/c.txt")
    path._origin = "root.x.y.z"
    path._consumer = "root.p.q.r"
    path._request_queue = Mock()
    path._response_queue = Mock()
    json_dump = json.dumps(path, cls=LightningJSONEncoder)
    assert "path" in json_dump
    # the replacement of \ is needed for Windows paths
    assert str(path).replace("\\", "\\\\") in json_dump
    assert "origin_name" in json_dump
    assert path._origin in json_dump
    assert "consumer_name" in json_dump
    assert path._consumer in json_dump


def test_path_to_dict_from_dict():
    path = Path("a/b/c.txt")
    path._origin = "root.x.y.z"
    path._consumer = "root.p.q.r"
    path._request_queue = Mock()
    path._response_queue = Mock()
    path_dict = path.to_dict()
    same_path = Path.from_dict(path_dict)
    assert same_path == path
    assert same_path._origin == path._origin
    assert same_path._consumer == path._consumer
    assert same_path._request_queue is None
    assert same_path._response_queue is None
    assert same_path._metadata == path._metadata


def test_path_attach_work():
    """Test that attaching a path to a LighitningWork will make the Work either the origin or a consumer."""
    path = Path()
    assert path._origin is None
    work1 = EmptyWork()
    work2 = EmptyWork()
    work3 = EmptyWork()
    path._attach_work(work=work1)
    assert path._origin is work1
    # path already has an owner
    path._attach_work(work=work2)
    assert path._origin is work1
    assert path._consumer is work2

    # path gets a new consumer
    path._attach_work(work=work3)
    assert path._origin is work1
    assert path._consumer is work3


def test_path_attach_queues():
    path = Path()
    request_queue = Mock()
    response_queue = Mock()
    path._attach_queues(request_queue=request_queue, response_queue=response_queue)
    assert path._request_queue is request_queue
    assert path._response_queue is response_queue


@pytest.mark.parametrize("cls", [LightningFlow, LightningWork])
def test_path_in_flow_and_work(cls, tmpdir):
    class PathComponent(cls):
        def __init__(self):
            super().__init__()
            self.path_one = Path("a", "b")
            self.path_one = Path("a", "b", "c")
            self.path_two = Path(tmpdir) / "write.txt"

        def run(self):
            self.path_one = self.path_one / "d.txt"
            assert self.path_one == Path("a", "b", "c", "d.txt")
            with open(self.path_two, "w") as file:
                file.write("Hello")

    class RootFlow(LightningFlow):
        def __init__(self):
            super().__init__()
            self.path_component = PathComponent()

        def run(self):
            self.path_component.run()

    root = RootFlow()
    _ = LightningApp(root)  # create an app to convert all paths that got attached

    root.run()

    assert root.path_component.path_one == Path("a", "b", "c", "d.txt")
    assert root.path_component.path_one == pathlib.Path("a", "b", "c", "d.txt")
    if isinstance(root.path_component, LightningWork):
        assert root.path_component.path_one.origin_name == "root.path_component"
        assert root.path_component.path_one.consumer_name == "root.path_component"
    else:
        assert root.path_component.path_one._origin is None
        assert root.path_component.path_one._consumer is None
    with open(root.path_component.path_two) as fo:
        assert fo.readlines() == ["Hello"]


class SourceWork(LightningWork):
    def __init__(self, tmpdir):
        super().__init__(cache_calls=True)
        self.path = Path(tmpdir, "src.txt")
        assert self.path.origin_name == ""

    def run(self):
        with open(self.path, "w") as f:
            f.write("Hello from SourceWork")


class DestinationWork(LightningWork):
    def __init__(self, source_path):
        super().__init__(cache_calls=True)
        assert source_path.origin_name == "root.src_work"
        self.path = source_path
        assert self.path.origin_name == "root.src_work"
        self.other = Path("other")
        assert self.other.origin_name == ""

    def run(self):
        assert self.path.origin_name == "root.src_work"
        assert self.other.origin_name == "root.dst_work"
        # we are running locally, the file is already there (no transfer needed)
        self.path.get(overwrite=True)
        assert self.path.is_file()
        assert self.path.read_text() == "Hello from SourceWork"


class SourceToDestFlow(LightningFlow):
    def __init__(self, tmpdir):
        super().__init__()
        self.src_work = SourceWork(tmpdir)
        self.dst_work = DestinationWork(self.src_work.path)

    def run(self):
        self.src_work.run()
        if self.src_work.has_succeeded:
            self.dst_work.run()
        if self.dst_work.has_succeeded:
            self.stop()


@pytest.mark.skipif(sys.platform == "win32" or sys.platform == "darwin", reason="too slow on Windows or macOs")
def test_multiprocess_path_in_work_and_flow(tmpdir):
    root = SourceToDestFlow(tmpdir)
    app = LightningApp(root, log_level="debug")
    MultiProcessRuntime(app, start_server=False).dispatch()


class DynamicSourceToDestFlow(LightningFlow):
    def __init__(self, tmpdir):
        super().__init__()
        self.tmpdir = str(tmpdir)

    def run(self):
        if not hasattr(self, "src_work"):
            self.src_work = SourceWork(self.tmpdir)
        self.src_work.run()
        if self.src_work.has_succeeded:
            if not hasattr(self, "dst_work"):
                self.dst_work = DestinationWork(self.src_work.path)
            self.dst_work.run()
        if hasattr(self, "dst_work") and self.dst_work.has_succeeded:
            self.stop()


# FIXME(alecmerdler): This test is failing...
@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut exception")
@pytest.mark.xfail(strict=False, reason="hanging...")
def test_multiprocess_path_in_work_and_flow_dynamic(tmpdir):
    root = DynamicSourceToDestFlow(tmpdir)
    app = LightningApp(root)
    MultiProcessRuntime(app).dispatch()


class RunPathFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.src_work = PathSourceWork()
        self.run_work = RunPathWork(cache_calls=True)

    def run(self):
        self.src_work.run()
        assert self.src_work.src_path_0.origin_name == "root.src_work"
        assert self.src_work.src_path_0.consumer_name == "root.src_work"

        # local_path is not attached to any Work
        local_path_0 = Path("local", "file_0.txt")
        local_path_1 = Path("local", "file_1.txt")
        assert local_path_0.origin_name is None
        assert local_path_0.consumer_name is None

        nested_local_path = (99, {"nested": local_path_1})
        nested_kwarg_path = ["x", (self.src_work.src_path_1,)]

        # TODO: support returning a path from run()
        self.run_work.run(
            self.src_work.src_path_0,
            local_path_0,
            nested_local_path,
            kwarg_path=local_path_1,
            nested_kwarg_path=nested_kwarg_path,
        )
        sleep(1)
        self.stop()


class PathSourceWork(EmptyWork):
    def __init__(self):
        super().__init__()
        self.src_path_0 = Path("src", "file_0.txt")
        self.src_path_1 = Path("src", "file_1.txt")


class RunPathWork(LightningWork):
    def run(self, src_path_0, local_path_0, nested_local_path, kwarg_path=None, nested_kwarg_path=None):
        all_paths = []

        # src_path_0 has an origin which must be preserved, this work becomes consumer
        assert str(src_path_0) == os.path.join("src", "file_0.txt")
        assert src_path_0.origin_name == "root.src_work"
        all_paths.append(src_path_0)

        # local_path_0 had no origin, this work becomes both the origin and the consumer
        assert str(local_path_0) == os.path.join("local", "file_0.txt")
        assert local_path_0.origin_name is None
        assert local_path_0.consumer_name is None
        all_paths.append(local_path_0)

        # nested_local_path is a nested container that contains a Path
        assert str(nested_local_path[1]["nested"]) == os.path.join("local", "file_1.txt")
        assert nested_local_path[1]["nested"].origin_name is None
        assert nested_local_path[1]["nested"].consumer_name is None
        all_paths.append(nested_local_path[1]["nested"])

        # keywoard arguments can also contain Paths
        assert str(kwarg_path) == os.path.join("local", "file_1.txt")
        assert kwarg_path.origin_name is None
        assert kwarg_path.consumer_name is None
        all_paths.append(kwarg_path)

        assert str(nested_kwarg_path[1][0]) == os.path.join("src", "file_1.txt")
        assert nested_kwarg_path[1][0].origin_name == "root.src_work"
        all_paths.append(nested_kwarg_path[1][0])

        all(p._request_queue == self._request_queue for p in all_paths)
        all(p._response_queue == self._response_queue for p in all_paths)
        all(p.consumer_name == self.name == "root.run_work" for p in all_paths)


def test_path_as_argument_to_run_method():
    """Test that Path objects can be passed as arguments to the run() method of a Work in various ways such that the
    origin, consumer and queues get automatically attached."""
    root = RunPathFlow()
    app = LightningApp(root)
    MultiProcessRuntime(app, start_server=False).dispatch()


def test_path_get_errors(tmpdir):
    with _context("work"):
        with pytest.raises(
            RuntimeError, match="Trying to get the file .* but the path is not attached to a LightningApp"
        ):
            Path().get()

        with pytest.raises(
            RuntimeError, match="Trying to get the file .* but the path is not attached to a LightningWork"
        ):
            path = Path()
            path._attach_queues(Mock(), Mock())
            path.get()

        with pytest.raises(FileExistsError, match="The file or folder .* exists locally. Pass `overwrite=True"):
            path = Path(tmpdir)
            path._attach_queues(Mock(), Mock())
            path._attach_work(Mock())
            path.get()


class SourceOverwriteWork(LightningWork):
    def __init__(self, tmpdir):
        super().__init__(raise_exception=True)
        self.path = Path(tmpdir, "folder")

    def run(self):
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "file.txt").touch()
        assert self.path.exists_local()


class DestinationOverwriteWork(LightningWork):
    def __init__(self, source_path):
        super().__init__(raise_exception=True)
        self.path = source_path

    def run(self):
        assert self.path.exists()
        with mock.patch("lightning.app.storage.path.shutil") as shutil_mock:
            self.path.get(overwrite=True)
        shutil_mock.rmtree.assert_called_with(self.path)
        assert self.path.exists()
        assert (self.path / "file.txt").exists()


class OverwriteFolderFlow(LightningFlow):
    def __init__(self, tmpdir):
        super().__init__()
        self.src_work = SourceOverwriteWork(tmpdir)
        self.dst_work = DestinationOverwriteWork(self.src_work.path)

    def run(self):
        self.src_work.run()
        if self.src_work.has_succeeded:
            self.dst_work.run()
        if self.dst_work.has_succeeded:
            self.stop()


def test_path_get_overwrite(tmpdir):
    """Test that .get(overwrite=True) overwrites the entire directory and replaces all files."""
    root = OverwriteFolderFlow(tmpdir)
    app = LightningApp(root, log_level="debug")
    MultiProcessRuntime(app, start_server=False).dispatch()


def test_path_get_error_in_flow_context():
    with pytest.raises(RuntimeError, match=escape("`Path.get()` can only be called from within the `run()`")), _context(
        "flow"
    ):
        Path().get()


def test_path_response_with_exception(tmpdir):
    request_queue = _MockQueue()
    response_queue = _MockQueue()
    path = Path(tmpdir / "file.txt")
    path._attach_queues(request_queue, response_queue)
    path._origin = "origin"
    path._consumer = "consumer"

    # simulate that a response will come with an exception raised
    response_queue.put(
        _GetResponse(
            source="origin",
            path=str(tmpdir / "file.txt"),
            hash=path.hash,
            destination="consumer",
            exception=OSError("Something went wrong"),
            name="",
        )
    )

    with pytest.raises(
        RuntimeError, match="An exception was raised while trying to transfer the contents at"
    ), _context("work"):
        path.get()


def test_path_response_not_matching_reqeuest(tmpdir):
    request_queue = _MockQueue()
    response_queue = _MockQueue()
    path = Path(tmpdir / "file.txt")
    path._attach_queues(request_queue, response_queue)
    path._origin = "origin"
    path._consumer = "consumer"

    # simulate a response that has a different owner than the request had
    response = _GetResponse(
        source="other_origin", path=str(tmpdir / "file.txt"), hash=path.hash, destination="consumer", name=""
    )

    response_queue.put(response)
    with pytest.raises(
        RuntimeError, match="Tried to get the file .* but received a response for a request it did not send."
    ):
        path.get()

    # simulate a response that has a different hash than the request had
    assert len(response_queue) == 0
    response.path = str(path)
    response.hash = "other_hash"
    response_queue.put(response)
    with pytest.raises(
        RuntimeError, match="Tried to get the file .* but received a response for a request it did not send."
    ):
        path.get()


def test_path_exists(tmpdir):
    """Test that the Path.exists() behaves as expected: First it should check if the file exists locally, and if not,
    send a message to the orchestrator to eventually check the existenc on the origin Work."""
    # Local Path (no Work queues attached)
    assert not Path("file").exists()
    assert Path(tmpdir).exists()
    with open(tmpdir / "file", "w"):
        assert Path(tmpdir / "file").exists()

    # A local path that exists
    path = Path(tmpdir)
    path.exists_remote = Mock()
    path.exists_local = Mock(return_value=True)
    assert path.exists() is True
    path.exists_local.assert_called_once()
    path.exists_remote.assert_not_called()  # don't check remotely

    # A local path that does not exist, but has no Work attached
    path = Path("not-exists.txt")
    path.exists_local = Mock(return_value=False)
    path.exists_remote = Mock()
    assert not path.exists()
    path.exists_local.assert_called_once()
    path.exists_remote.assert_not_called()  # don't check remotely

    # A local path that does not exist, but it exists remotely
    path = Path("exists-remotely-only.txt")
    path.exists_local = Mock(return_value=False)
    path.exists_remote = Mock(return_value=True)
    path._origin = "origin"
    assert path.exists()
    path.exists_local.assert_called_once()
    path.exists_remote.assert_called_once()  # check remotely


def test_path_exists_local(tmpdir):
    assert not Path("file").exists_local()
    assert Path(tmpdir).exists_local()
    with open(tmpdir / "file", "w"):
        assert Path(tmpdir / "file").exists_local()


def test_path_exists_remote(tmpdir):
    path = Path(tmpdir / "not-attached.txt")
    with pytest.raises(RuntimeError, match="the path is not attached to a LightningWork"):
        path.exists_remote()

    # If Path does not exist locally, ask the orchestrator
    request_queue = _MockQueue()
    response_queue = _MockQueue()
    path = Path(tmpdir / "not-exists.txt")
    path._attach_queues(request_queue, response_queue)
    path._origin = "origin"
    path._consumer = "consumer"

    # Put the response into the queue to simulate the orchestrator responding
    response_queue.put(_ExistsResponse(source=path.origin_name, path=str(path), name="", hash="123", exists=False))
    assert not path.exists_remote()
    assert request_queue.get()

    response_queue.put(_ExistsResponse(source=path.origin_name, path=str(path), name="", hash="123", exists=True))
    assert path.exists_remote()
    assert request_queue.get()


def test_artifacts_path():
    work = Mock()
    work.name = "root.flow.work"
    assert _artifacts_path(work) == _shared_storage_path() / "artifacts" / "root.flow.work"


@pytest.mark.skipif(not _is_s3fs_available(), reason="This test requires s3fs.")
@mock.patch.dict(os.environ, {"LIGHTNING_BUCKET_ENDPOINT_URL": "a"})
@mock.patch.dict(os.environ, {"LIGHTNING_BUCKET_NAME": "b"})
@mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_APP_ID": "e"})
def test_filesystem(monkeypatch):
    from lightning.app.storage import path

    mock = MagicMock()
    monkeypatch.setattr(path, "S3FileSystem", mock)
    fs = _filesystem()
    assert fs == mock()


class TestSharedStoragePath(TestCase):
    @mock.patch.dict(os.environ, {"LIGHTNING_STORAGE_PATH": "test-bucket/lightningapps/test-project/test-app"})
    def test_shared_storage_path_storage_path_set(self):
        assert pathlib.Path("test-bucket/lightningapps/test-project/test-app") == _shared_storage_path()

    @mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_APP_ID": "test-app", "LIGHTNING_BUCKET_NAME": "test-bucket"})
    def test_shared_storage_path_bucket_and_app_id_set(self):
        assert pathlib.Path("test-bucket/lightningapps/test-app") == _shared_storage_path()

    @mock.patch.dict(os.environ, {"SHARED_MOUNT_DIRECTORY": "test-app/.shared"})
    def test_shared_storage_path_mount_directory_set(self):
        assert _shared_storage_path().match("*/test-app/.shared")

    def test_shared_storage_path_no_envvars_set(self):
        assert _shared_storage_path().match("*/.shared")
