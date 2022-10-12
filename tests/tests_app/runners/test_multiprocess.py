from unittest import mock
from unittest.mock import MagicMock, Mock

from lightning_app import LightningApp, LightningFlow, LightningWork
from lightning_app.frontend import StaticWebFrontend, StreamlitFrontend
from lightning_app.runners import MultiProcessRuntime
from lightning_app.testing.helpers import EmptyFlow
from lightning_app.utilities.component import _get_context


def _streamlit_render_fn():
    pass


class StreamlitFlow(LightningFlow):
    def run(self):
        self._exit()

    def configure_layout(self):
        frontend = StreamlitFrontend(render_fn=_streamlit_render_fn)
        frontend.start_server = Mock()
        frontend.stop_server = Mock()
        return frontend


class WebFlow(LightningFlow):
    def run(self):
        self._exit()

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
        self._exit()


@mock.patch("lightning_app.runners.multiprocess.find_free_network_port")
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


class ContxtFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = ContextWork()
        assert _get_context() is None

    def run(self):
        assert _get_context().value == "flow"
        self.work.run()
        assert _get_context().value == "flow"
        self._exit()


def test_multiprocess_runtime_sets_context():
    """Test that the runtime sets the global variable COMPONENT_CONTEXT in Flow and Work."""
    MultiProcessRuntime(LightningApp(ContxtFlow())).dispatch()


@mock.patch("lightning_app.runners.multiprocess.constants", MagicMock(ENABLE_APP_CHECKPOINT=True))
def test_dispatch_loads_app_from_checkpoint():

    app = LightningApp(EmptyFlow())
    mp_runtime = MultiProcessRuntime(app, checkpoint="test_checkpoint")
    app.load_app_state_from_checkpoint = Mock()
    app._run = Mock()

    mp_runtime.dispatch()
    app.load_app_state_from_checkpoint.assert_called_once_with("test_checkpoint")
