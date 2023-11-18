import os
import sys
from unittest import mock
from unittest.mock import Mock

import pytest
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.core import constants
from lightning.app.frontend import StaticWebFrontend, StreamlitFrontend
from lightning.app.runners import MultiProcessRuntime
from lightning.app.utilities.component import _get_context
from lightning.app.utilities.imports import _IS_WINDOWS


def _streamlit_render_fn():
    pass


class StreamlitFlow(LightningFlow):
    def run(self):
        self.stop()

    def configure_layout(self):
        frontend = StreamlitFrontend(render_fn=_streamlit_render_fn)
        frontend.start_server = Mock()
        frontend.stop_server = Mock()
        return frontend


class WebFlow(LightningFlow):
    def run(self):
        self.stop()

    def configure_layout(self):
        frontend = StaticWebFrontend(serve_dir="a/b/c")
        frontend.start_server = Mock()
        frontend.stop_server = Mock()
        return frontend


class StartFrontendServersTestFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.flow0 = StreamlitFlow()
        self.flow1 = WebFlow()

    def run(self):
        self.stop()


@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut exception")
@pytest.mark.xfail(strict=False, reason="hanging with timeout")  # fixme
@pytest.mark.parametrize(
    ("cloudspace_host", "port", "expected_host", "expected_target"),
    [
        (None, 7000, "localhost", "http://localhost:7000"),
        ("test.lightning.ai", 7000, "0.0.0.0", "https://7000-test.lightning.ai"),  # noqa: S104
    ],
)
@mock.patch("lightning.app.runners.multiprocess.find_free_network_port")
def test_multiprocess_starts_frontend_servers(
    mock_find_free_network_port, monkeypatch, cloudspace_host, port, expected_host, expected_target
):
    """Test that the MultiProcessRuntime starts the servers for the frontends in each LightningFlow."""

    monkeypatch.setattr(constants, "LIGHTNING_CLOUDSPACE_HOST", cloudspace_host)
    mock_find_free_network_port.return_value = port

    root = StartFrontendServersTestFlow()
    app = LightningApp(root)
    MultiProcessRuntime(app).dispatch()

    app.frontends[root.flow0.name].start_server.assert_called_once()
    assert app.frontends[root.flow0.name].start_server.call_args.kwargs["host"] == expected_host

    app.frontends[root.flow1.name].start_server.assert_called_once()
    assert app.frontends[root.flow1.name].start_server.call_args.kwargs["host"] == expected_host

    assert app.frontends[root.flow0.name].flow._layout["target"] == f"{expected_target}/{root.flow0.name}"
    assert app.frontends[root.flow1.name].flow._layout["target"] == f"{expected_target}/{root.flow1.name}"

    app.frontends[root.flow0.name].stop_server.assert_called_once()
    app.frontends[root.flow1.name].stop_server.assert_called_once()


class ContextWork(LightningWork):
    def __init__(self):
        super().__init__()

    def run(self):
        assert _get_context().value == "work"


class ContextFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = ContextWork()
        assert _get_context() is None

    def run(self):
        assert _get_context().value == "flow"
        self.work.run()
        assert _get_context().value == "flow"
        self.stop()


@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut exception")
@pytest.mark.xfail(strict=False, reason="hanging with timeout")  # fixme
def test_multiprocess_runtime_sets_context():
    """Test that the runtime sets the global variable COMPONENT_CONTEXT in Flow and Work."""
    MultiProcessRuntime(LightningApp(ContextFlow())).dispatch()


@pytest.mark.parametrize(
    ("env", "expected_url"),
    [
        ({}, "http://127.0.0.1:7501/view"),
        ({"APP_SERVER_HOST": "http://test"}, "http://test"),
    ],
)
@pytest.mark.skipif(sys.platform == "win32", reason="hanging with timeout")
def test_get_app_url(env, expected_url):
    with mock.patch.dict(os.environ, env):
        assert MultiProcessRuntime._get_app_url() == expected_url
