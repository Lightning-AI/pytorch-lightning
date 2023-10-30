import os
import pathlib
import pickle
from copy import deepcopy
from unittest import mock
from unittest.mock import Mock

import pytest
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.runners.multiprocess import MultiProcessRuntime
from lightning.app.storage.payload import Payload
from lightning.app.storage.requests import _GetRequest


def test_payload_copy():
    """Test that Payload creates an exact copy when passing a Payload instance to the constructor."""
    payload = Payload(None)
    payload._origin = "origin"
    payload._consumer = "consumer"
    payload._request_queue = "MockQueue"
    payload._response_queue = "MockQueue"
    payload_copy = deepcopy(payload)
    assert payload_copy._origin == payload._origin
    assert payload_copy._consumer == payload._consumer
    assert payload_copy._request_queue == payload._request_queue
    assert payload_copy._response_queue == payload._response_queue


def test_payload_pickable():
    payload = Payload("MyObject")
    payload._origin = "root.x.y.z"
    payload._consumer = "root.p.q.r"
    payload._name = "var_a"
    loaded = pickle.loads(pickle.dumps(payload))

    assert isinstance(loaded, Payload)
    assert loaded._origin == payload._origin
    assert loaded._consumer == payload._consumer
    assert loaded._name == payload._name
    assert loaded._request_queue is None
    assert loaded._response_queue is None


def test_path_attach_queues():
    path = Payload(None)
    request_queue = Mock()
    response_queue = Mock()
    path._attach_queues(request_queue=request_queue, response_queue=response_queue)
    assert path._request_queue is request_queue
    assert path._response_queue is response_queue


class Work(LightningWork):
    def __init__(self):
        super().__init__()
        self.var_a = Payload(None)

    def run(self):
        pass


def test_payload_in_init():
    with pytest.raises(
        AttributeError, match="The Payload object should be set only within the run method of the work."
    ):
        Work()


class WorkRun(LightningWork):
    def __init__(self, tmpdir):
        super().__init__()
        self.var_a = None
        self.tmpdir = tmpdir

    def run(self):
        self.var_a = Payload("something")
        assert self.var_a.name == "var_a"
        assert self.var_a._origin == "root.a"
        assert self.var_a.hash == "9bd514ad51fc33d895c50657acd0f0582301cf3e"
        source_path = pathlib.Path(self.tmpdir, self.var_a.name)
        assert not source_path.exists()
        response = self.var_a._handle_get_request(
            self,
            _GetRequest(
                name="var_a",
                hash=self.var_a.hash,
                source="root.a",
                path=str(source_path),
                destination="root",
            ),
        )
        assert source_path.exists()
        assert self.var_a.load(str(source_path)) == "something"
        assert not response.exception


def test_payload_in_run(tmpdir):
    work = WorkRun(str(tmpdir))
    work._name = "root.a"
    work.run()


class Sender(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.value_all = None
        self.value_b = None
        self.value_c = None

    def run(self):
        self.value_all = Payload(["A", "B", "C"])
        self.value_b = Payload("B")
        self.value_c = Payload("C")


class WorkReceive(LightningWork):
    def __init__(self, expected):
        super().__init__(parallel=True)
        self.expected = expected

    def run(self, generated):
        assert generated.value == self.expected


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.sender = Sender()
        self.receiver_all = WorkReceive(["A", "B", "C"])
        self.receiver_b = WorkReceive("B")
        self.receiver_c = WorkReceive("C")

    def run(self):
        self.sender.run()
        if self.sender.value_all:
            self.receiver_all.run(self.sender.value_all)
        if self.sender.value_b:
            self.receiver_b.run(self.sender.value_b)
        if self.sender.value_c:
            self.receiver_c.run(self.sender.value_c)
        if self.receiver_all.has_succeeded and self.receiver_b.has_succeeded and self.receiver_c.has_succeeded:
            self.stop()


@pytest.mark.xfail(strict=False, reason="flaky")
def test_payload_works(tmpdir):
    """This tests validates the payload api can be used to transfer return values from a work to another."""
    with mock.patch("lightning.app.storage.path._storage_root_dir", return_value=pathlib.Path(tmpdir)):
        app = LightningApp(Flow(), log_level="debug")
        MultiProcessRuntime(app, start_server=False).dispatch()

    os.remove("value_all")
    os.remove("value_b")
    os.remove("value_c")
