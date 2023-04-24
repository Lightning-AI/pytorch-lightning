from unittest import mock
from unittest.mock import MagicMock, Mock

from lightning.app import LightningApp, LightningWork
from lightning.app.runners.backends import MultiProcessingBackend


@mock.patch("lightning.app.core.app.AppStatus")
@mock.patch("lightning.app.runners.backends.mp_process.multiprocessing")
def test_backend_create_work_with_set_start_method(multiprocessing_mock, *_):
    backend = MultiProcessingBackend(entrypoint_file="fake.py")
    work = Mock(spec=LightningWork)
    work._start_method = "test_start_method"

    app = LightningApp(work)
    app.caller_queues = MagicMock()
    app.delta_queue = MagicMock()
    app.readiness_queue = MagicMock()
    app.error_queue = MagicMock()
    app.request_queues = MagicMock()
    app.response_queues = MagicMock()
    app.copy_request_queues = MagicMock()
    app.copy_response_queues = MagicMock()
    app.flow_to_work_delta_queues = MagicMock()

    backend.create_work(app=app, work=work)
    multiprocessing_mock.get_context.assert_called_with("test_start_method")
    multiprocessing_mock.get_context().Process().start.assert_called_once()
