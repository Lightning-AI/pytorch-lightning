import os
from time import sleep, time

import pytest
from lightning.app import LightningWork
from lightning.app.core.queues import QueuingSystem
from lightning.app.testing.helpers import _RunIf
from lightning.app.utilities.imports import _is_docker_available
from lightning.app.utilities.load_app import load_app_from_file
from lightning.app.utilities.packaging.docker import DockerRunner
from lightning.app.utilities.redis import check_if_redis_running


@pytest.mark.xfail(strict=False, reason="FIXME (tchaton)")
@pytest.mark.skipif(not _is_docker_available(), reason="docker is required for this test.")
@pytest.mark.skipif(not check_if_redis_running(), reason="redis is required for this test.")
@_RunIf(skip_windows=True)
def test_docker_runner():
    """This test validates that the lightning run work is executable within a container and deltas are sent back
    through the Redis caller_queue."""
    queues = QueuingSystem.REDIS
    queue_id = f"test_docker_runner_{str(int(time()))}"
    app_file = os.path.join(os.path.dirname(__file__), "projects/dock/app.py")
    app = load_app_from_file(app_file)
    work: LightningWork = app.root.work

    call_hash = "run:fe3fa0f34fc1317e152e5afb023332995392071046f1ea51c34c7c9766e3676c"
    work._calls[call_hash] = {
        "args": (),
        "kwargs": {},
        "call_hash": call_hash,
        "run_started_counter": 1,
        "statuses": [],
    }

    # The script_path needs to be relative to the container.
    docker_runner = DockerRunner(
        "/home/tests/utilities/packaging/projects/dock/app.py", work, queue_id, create_base=True
    )
    docker_runner.run()

    caller_queue = queues.get_caller_queue(work_name=work.name, queue_id=queue_id)
    caller_queue.put({
        "args": (),
        "kwargs": {},
        "call_hash": call_hash,
        "state": work.state,
    })
    delta_queue = queues.get_delta_queue(queue_id=queue_id)
    delta_1 = delta_queue.get()
    delta_2 = delta_queue.get()
    delta_3 = delta_queue.get()

    def get_item(delta):
        return delta.delta.to_dict()["iterable_item_added"]

    assert delta_1.id == "root.work"
    assert delta_2.id == "root.work"
    assert delta_3.id == "root.work"
    assert get_item(delta_1)[f"root['calls']['{call_hash}']['statuses'][0]"]["stage"] == "starting"
    assert delta_2.delta.to_dict()["type_changes"]["root['vars']['message']"]["new_value"] == "hello world!"
    assert get_item(delta_3)[f"root['calls']['{call_hash}']['statuses'][1]"]["stage"] == "succeeded"
    sleep(1)
    assert "Starting WorkRunner" in docker_runner.get_container_logs()
    docker_runner.kill()
