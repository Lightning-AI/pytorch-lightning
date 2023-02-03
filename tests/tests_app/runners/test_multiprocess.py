import os
from unittest import mock
from unittest.mock import Mock

import pytest

from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.frontend import StaticWebFrontend, StreamlitFrontend
from lightning.app.runners import MultiProcessRuntime
from lightning.app.utilities.component import _get_context


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


@mock.patch("lightning.app.runners.multiprocess.find_free_network_port")
def test_multiprocess_starts_frontend_servers(*_):
    """Test that the MultiProcessRuntime starts the servers for the frontends in each LightningFlow."""
    root = StartFrontendServersTestFlow()
    app = LightningApp(root)
    MultiProcessRuntime(app).dispatch()

    app.frontends[root.flow0.name].start_server.assert_called_once()
    app.frontends[root.flow1.name].start_server.assert_called_once()

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


def test_multiprocess_runtime_sets_context():
    """Test that the runtime sets the global variable COMPONENT_CONTEXT in Flow and Work."""
    MultiProcessRuntime(LightningApp(ContextFlow())).dispatch()


@pytest.mark.parametrize(
    "env,expected_url",
    [
        ({}, "http://127.0.0.1:7501/view"),
        ({"APP_SERVER_HOST": "http://test"}, "http://test"),
    ],
)
def test_get_app_url(env, expected_url):
    with mock.patch.dict(os.environ, env):
        assert MultiProcessRuntime._get_app_url() == expected_url
