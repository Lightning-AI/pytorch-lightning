import os
import signal
import time
from functools import partial
from multiprocessing import Process
from pathlib import Path
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
from click.testing import CliRunner
from lightning.app.cli.lightning_cli_launch import run_flow, run_flow_and_servers, run_frontend, run_server
from lightning.app.core.queues import QueuingSystem
from lightning.app.frontend.web import StaticWebFrontend
from lightning.app.launcher import launcher
from lightning.app.runners.runtime import load_app_from_file
from lightning.app.testing.helpers import EmptyWork, _RunIf
from lightning.app.utilities.app_commands import run_app_commands
from lightning.app.utilities.network import find_free_network_port

from tests_app import _PROJECT_ROOT

_FILE_PATH = os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py")


def test_run_frontend(monkeypatch):
    """Test that the CLI can be used to start the frontend server of a particular LightningFlow using the cloud
    dispatcher.

    This CLI call is made by Lightning AI and is not meant to be invoked by the user directly.

    """
    runner = CliRunner()

    port = find_free_network_port()

    start_server_mock = Mock()
    monkeypatch.setattr(StaticWebFrontend, "start_server", start_server_mock)

    result = runner.invoke(
        run_frontend,
        [
            str(Path(__file__).parent / "launch_data" / "app_v0" / "app.py"),
            "--flow-name",
            "root.aas",
            "--host",
            "localhost",
            "--port",
            port,
        ],
    )
    assert result.exit_code == 0
    start_server_mock.assert_called_once()
    start_server_mock.assert_called_with("localhost", port)


class MockRedisQueue:
    _MOCKS = {}

    def __init__(self, name: str, default_timeout: float):
        self.name = name
        self.default_timeout = default_timeout
        self.queue = [None]  # adding a dummy element.

        self._MOCKS[name] = MagicMock()

    def put(self, item):
        self._MOCKS[self.name].put(item)
        self.queue.put(item)

    def get(self, timeout: int = None):
        self._MOCKS[self.name].get(timeout=timeout)
        return self.queue.pop(0)

    @property
    def is_running(self):
        self._MOCKS[self.name].is_running()
        return True


@mock.patch("lightning.app.core.queues.RedisQueue", MockRedisQueue)
@mock.patch("lightning.app.launcher.launcher.check_if_redis_running", MagicMock(return_value=True))
@mock.patch("lightning.app.launcher.launcher.start_server")
def test_run_server(start_server_mock):
    runner = CliRunner()
    result = runner.invoke(
        run_server,
        [
            _FILE_PATH,
            "--queue-id",
            "1",
            "--host",
            "http://127.0.0.1:7501/view",
            "--port",
            "6000",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    start_server_mock.assert_called_once_with(
        host="http://127.0.0.1:7501/view",
        port=6000,
        api_publish_state_queue=ANY,
        api_delta_queue=ANY,
        api_response_queue=ANY,
        spec=ANY,
        apis=ANY,
    )
    kwargs = start_server_mock._mock_call_args.kwargs
    assert isinstance(kwargs["api_publish_state_queue"], MockRedisQueue)
    assert kwargs["api_publish_state_queue"].name.startswith("1")
    assert isinstance(kwargs["api_delta_queue"], MockRedisQueue)
    assert kwargs["api_delta_queue"].name.startswith("1")


def mock_server(should_catch=False, sleep=1000):
    if should_catch:

        def _sigterm_handler(*_):
            time.sleep(100)

        signal.signal(signal.SIGTERM, _sigterm_handler)

    time.sleep(sleep)


def run_forever_process():
    while True:
        time.sleep(1)


def run_for_2_seconds_and_raise():
    time.sleep(2)
    raise RuntimeError("existing")


def exit_successfully_immediately():
    return


def start_servers(should_catch=False, sleep=1000):
    processes = [
        (
            "p1",
            launcher.start_server_in_process(target=partial(mock_server, should_catch=should_catch, sleep=sleep)),
        ),
        (
            "p2",
            launcher.start_server_in_process(target=partial(mock_server, sleep=sleep)),
        ),
        (
            "p3",
            launcher.start_server_in_process(target=partial(mock_server, sleep=sleep)),
        ),
    ]

    launcher.manage_server_processes(processes)


@_RunIf(skip_windows=True)
def test_manage_server_processes():
    p = Process(target=partial(start_servers, sleep=0.5))
    p.start()
    p.join()

    assert p.exitcode == 0

    p = Process(target=start_servers)
    p.start()
    p.join(0.5)
    p.terminate()
    p.join()

    assert p.exitcode in [-15, 0]

    p = Process(target=partial(start_servers, should_catch=True))
    p.start()
    p.join(0.5)
    p.terminate()
    p.join()

    assert p.exitcode in [-15, 1]


def start_processes(**functions):
    processes = []
    for name, fn in functions.items():
        processes.append((name, launcher.start_server_in_process(fn)))
    launcher.manage_server_processes(processes)


@_RunIf(skip_windows=True)
@pytest.mark.flaky(reruns=3)
def test_manage_server_processes_one_process_gets_killed(capfd):
    functions = {"p1": run_forever_process, "p2": run_for_2_seconds_and_raise}
    p = Process(target=start_processes, kwargs=functions)
    p.start()

    for _ in range(40):
        time.sleep(1)
        if p.exitcode is not None:
            break
    assert p.exitcode == 1
    captured = capfd.readouterr()
    assert (
        "Found dead components with non-zero exit codes, exiting execution!!! Components: \n"
        "| Name | Exit Code |\n|------|-----------|\n| p2   | 1         |\n" in captured.out
    )


@_RunIf(skip_windows=True)
def test_manage_server_processes_all_processes_exits_with_zero_exitcode(capfd):
    functions = {
        "p1": exit_successfully_immediately,
        "p2": exit_successfully_immediately,
    }
    p = Process(target=start_processes, kwargs=functions)
    p.start()

    for _ in range(40):
        time.sleep(1)
        if p.exitcode is not None:
            break
    assert p.exitcode == 0
    captured = capfd.readouterr()
    assert "All the components are inactive with exitcode 0. Exiting execution!!!" in captured.out


@mock.patch("lightning.app.launcher.launcher.StorageOrchestrator", MagicMock())
@mock.patch("lightning.app.core.queues.RedisQueue", MockRedisQueue)
@mock.patch("lightning.app.launcher.launcher.manage_server_processes", Mock())
def test_run_flow_and_servers(monkeypatch):
    runner = CliRunner()

    start_server_mock = Mock()
    monkeypatch.setattr(launcher, "start_server_in_process", start_server_mock)

    runner.invoke(
        run_flow_and_servers,
        [
            str(Path(__file__).parent / "launch_data" / "app_v0" / "app.py"),
            "--base-url",
            "https://some.url",
            "--queue-id",
            "1",
            "--host",
            "http://127.0.0.1:7501/view",
            "--port",
            6000,
            "--flow-port",
            "root.aas",
            6001,
            "--flow-port",
            "root.bbs",
            6002,
        ],
        catch_exceptions=False,
    )

    start_server_mock.assert_called()
    assert start_server_mock.call_count == 4


@mock.patch("lightning.app.core.queues.RedisQueue", MockRedisQueue)
@mock.patch("lightning.app.launcher.launcher.WorkRunner")
def test_run_work(mock_work_runner, monkeypatch):
    run_app_commands(_FILE_PATH)
    app = load_app_from_file(_FILE_PATH)
    names = [w.name for w in app.works]

    mocked_queue = MagicMock()
    mocked_queue.get.return_value = EmptyWork()
    monkeypatch.setattr(
        QueuingSystem,
        "get_work_queue",
        MagicMock(return_value=mocked_queue),
    )

    assert names == [
        "root.flow_a_1.work_a",
        "root.flow_a_2.work_a",
        "root.flow_b.work_b",
    ]

    for name in names:
        launcher.run_lightning_work(
            file=_FILE_PATH,
            work_name=name,
            queue_id="1",
        )
        kwargs = mock_work_runner._mock_call_args.kwargs
        assert isinstance(kwargs["work"], EmptyWork)
        assert kwargs["work_name"] == name
        assert isinstance(kwargs["caller_queue"], MockRedisQueue)
        assert kwargs["caller_queue"].name.startswith("1")
        assert isinstance(kwargs["delta_queue"], MockRedisQueue)
        assert kwargs["delta_queue"].name.startswith("1")
        assert isinstance(kwargs["readiness_queue"], MockRedisQueue)
        assert kwargs["readiness_queue"].name.startswith("1")
        assert isinstance(kwargs["error_queue"], MockRedisQueue)
        assert kwargs["error_queue"].name.startswith("1")
        assert isinstance(kwargs["request_queue"], MockRedisQueue)
        assert kwargs["request_queue"].name.startswith("1")
        assert isinstance(kwargs["response_queue"], MockRedisQueue)
        assert kwargs["response_queue"].name.startswith("1")
        assert isinstance(kwargs["copy_request_queue"], MockRedisQueue)
        assert kwargs["copy_request_queue"].name.startswith("1")
        assert isinstance(kwargs["copy_response_queue"], MockRedisQueue)
        assert kwargs["copy_response_queue"].name.startswith("1")

        MockRedisQueue._MOCKS["healthz"].is_running.assert_called()


@mock.patch("lightning.app.core.queues.QueuingSystem", MagicMock())
@mock.patch("lightning.app.launcher.launcher.StorageOrchestrator", MagicMock())
@mock.patch("lightning.app.LightningApp._run")
@mock.patch("lightning.app.launcher.launcher.CloudBackend")
def test_run_flow(mock_cloud_backend, mock_lightning_app_run):
    runner = CliRunner()

    base_url = "https://lightning.ai/me/apps"

    result = runner.invoke(
        run_flow,
        [_FILE_PATH, "--queue-id=1", f"--base-url={base_url}"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    mock_lightning_app_run.assert_called_once()
    assert len(mock_cloud_backend._mock_mock_calls) == 13
