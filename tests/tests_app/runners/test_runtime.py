import os
import signal
from unittest import mock

import pytest
from lightning.app.runners import cloud
from lightning.app.runners.runtime import dispatch
from lightning.app.runners.runtime_type import RuntimeType

from tests_app import _PROJECT_ROOT


@pytest.mark.parametrize(
    "runtime_type",
    [
        RuntimeType.MULTIPROCESS,
        RuntimeType.CLOUD,
    ],
)
@mock.patch("lightning.app.core.queues.QueuingSystem", mock.MagicMock())
@mock.patch("lightning.app.runners.backends.cloud.LightningClient", mock.MagicMock())
def test_dispatch(runtime_type, monkeypatch):
    """This test ensures the runtime dispatch method gets called when using dispatch."""
    monkeypatch.setattr(cloud, "CloudBackend", mock.MagicMock())

    with pytest.raises(FileNotFoundError, match="doesnt_exists.py"):
        dispatch(
            entrypoint_file=os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/doesnt_exists.py"),
            runtime_type=runtime_type,
            start_server=False,
        )

    runtime = runtime_type.get_runtime()
    dispath_method_path = f"{runtime.__module__}.{runtime.__name__}.dispatch"

    with mock.patch(dispath_method_path) as dispatch_mock_fn:
        dispatch(
            entrypoint_file=os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py"),
            runtime_type=runtime_type,
            start_server=False,
        )
        dispatch_mock_fn.assert_called_once()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
