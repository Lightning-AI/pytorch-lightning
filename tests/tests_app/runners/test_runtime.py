import signal
from unittest import mock

import pytest

from lightning_app.runners import cloud
from lightning_app.runners.runtime import dispatch
from lightning_app.runners.runtime_type import RuntimeType


@pytest.mark.parametrize(
    "runtime_type",
    [
        RuntimeType.SINGLEPROCESS,
        RuntimeType.MULTIPROCESS,
        RuntimeType.CLOUD,
    ],
)
@mock.patch("lightning_app.core.queues.QueuingSystem", mock.MagicMock())
@mock.patch("lightning_app.runners.backends.cloud.LightningClient", mock.MagicMock())
def test_dispatch(runtime_type, monkeypatch):
    """This test ensures the runtime dispatch method gets called when using dispatch."""

    entrypoint_file = "tests/core/scripts/doesnt_exists.py"

    monkeypatch.setattr(cloud, "CloudBackend", mock.MagicMock())

    with pytest.raises(FileNotFoundError, match=entrypoint_file):
        dispatch(
            entrypoint_file=entrypoint_file,
            runtime_type=runtime_type,
            start_server=False,
        )

    runtime = runtime_type.get_runtime()
    dispath_method_path = f"{runtime.__module__}.{runtime.__name__}.dispatch"

    with mock.patch(dispath_method_path) as dispatch_mock_fn:
        dispatch(
            entrypoint_file="tests/core/scripts/app_metadata.py",
            runtime_type=runtime_type,
            start_server=False,
        )
        dispatch_mock_fn.assert_called_once()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
