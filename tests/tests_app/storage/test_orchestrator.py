from unittest.mock import MagicMock

from lightning.app.storage.orchestrator import StorageOrchestrator
from lightning.app.storage.requests import _GetRequest, _GetResponse
from lightning.app.testing.helpers import _MockQueue
from lightning.app.utilities.enum import WorkStageStatus


def test_orchestrator():
    """Simulate orchestration when Work B requests a file from Work A."""
    request_queues = {"work_a": _MockQueue(), "work_b": _MockQueue()}
    response_queues = {"work_a": _MockQueue(), "work_b": _MockQueue()}
    copy_request_queues = {"work_a": _MockQueue(), "work_b": _MockQueue()}
    copy_response_queues = {"work_a": _MockQueue(), "work_b": _MockQueue()}
    app = MagicMock()
    work = MagicMock()
    work.status.stage = WorkStageStatus.RUNNING
    app.get_component_by_name = MagicMock(return_value=work)

    orchestrator = StorageOrchestrator(
        app,
        request_queues=request_queues,
        response_queues=response_queues,
        copy_request_queues=copy_request_queues,
        copy_response_queues=copy_response_queues,
    )

    # test idle behavior when queues are empty
    orchestrator.run_once("work_a")
    orchestrator.run_once("work_b")
    assert not orchestrator.waiting_for_response

    # simulate Work B sending a request for a file in Work A
    request = _GetRequest(source="work_a", path="/a/b/c.txt", hash="", destination="", name="")
    request_queues["work_b"].put(request)
    orchestrator.run_once("work_a")
    assert not orchestrator.waiting_for_response
    orchestrator.run_once("work_b")

    # orchestrator is now waiting for a response for copier in Work A
    assert "work_b" in orchestrator.waiting_for_response
    assert len(request_queues["work_a"]) == 0
    assert request in copy_request_queues["work_a"]
    assert request.destination == "work_b"

    # simulate loop while waiting for new elements in the queues
    orchestrator.run_once("work_a")
    orchestrator.run_once("work_b")

    # edge case: `None` requests on the queue get ignored
    # TODO: Investigate how `None` values end up in the queue
    request_queues["work_a"].put(None)
    orchestrator.run_once("work_a")
    orchestrator.run_once("work_b")
    assert not request_queues["work_a"]._queue

    # simulate copier A confirms that the file is available on the shared volume
    response = _GetResponse(source="work_a", path="/a/b/c.txt", hash="", destination="work_b", name="")
    copy_request_queues["work_a"].get()
    copy_response_queues["work_a"].put(response)

    # orchestrator processes confirmation and confirms to the pending request from Work B
    orchestrator.run_once("work_a")
    assert len(copy_response_queues["work_a"]) == 0
    assert response in response_queues["work_b"]
    assert not orchestrator.waiting_for_response
    orchestrator.run_once("work_b")

    # simulate loop while waiting for new elements in the queues
    orchestrator.run_once("work_a")
    orchestrator.run_once("work_b")
    assert not orchestrator.waiting_for_response

    # simulate Work B receiving the confirmation that the file was copied
    response = response_queues["work_b"].get()
    assert response.source == "work_a"
    assert response.destination == "work_b"
    assert response.exception is None

    # all queues should be empty
    assert all(len(queue) == 0 for queue in request_queues.values())
    assert all(len(queue) == 0 for queue in response_queues.values())
    assert all(len(queue) == 0 for queue in copy_request_queues.values())
    assert all(len(queue) == 0 for queue in copy_response_queues.values())
