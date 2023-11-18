from re import escape
from unittest import mock
from unittest.mock import Mock

import pytest
from lightning.app import LightningApp, LightningFlow
from lightning.app.frontend.stream_lit import StreamlitFrontend
from lightning.app.frontend.web import StaticWebFrontend
from lightning.app.runners import MultiProcessRuntime
from lightning.app.testing.helpers import EmptyFlow
from lightning.app.utilities.imports import _IS_WINDOWS


@pytest.mark.parametrize("return_val", [1, None, set(), "string"])
def test_invalid_layout(return_val):
    class Root(EmptyFlow):
        def configure_layout(self):
            return return_val

    root = Root()
    with pytest.raises(TypeError, match=escape("The return value of configure_layout() in `Root`")):
        LightningApp(root)


def test_invalid_layout_missing_content_key():
    class Root(EmptyFlow):
        def configure_layout(self):
            return [{"name": "one"}]

    root = Root()
    with pytest.raises(
        ValueError, match=escape("A dictionary returned by `Root.configure_layout()` is missing a key 'content'.")
    ):
        LightningApp(root)


def test_invalid_layout_unsupported_content_value():
    class Root(EmptyFlow):
        def configure_layout(self):
            return [{"name": "one", "content": [1, 2, 3]}]

    root = Root()

    with pytest.raises(
        ValueError,
        match=escape("A dictionary returned by `Root.configure_layout()"),
    ):
        LightningApp(root)


class StreamlitFrontendFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        if self.counter > 2:
            self.stop()
        self.counter += 1

    def configure_layout(self):
        frontend = StreamlitFrontend(render_fn=_render_streamlit_fn)
        frontend.start_server = Mock()
        frontend.stop_server = Mock()
        return frontend


def _render_streamlit_fn():
    pass


class StaticWebFrontendFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        if self.counter > 2:
            self.stop()
        self.counter += 1

    def configure_layout(self):
        frontend = StaticWebFrontend(serve_dir="a/b/c")
        frontend.start_server = Mock()
        frontend.stop_server = Mock()
        return frontend


@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut exception")
@pytest.mark.xfail(strict=False, reason="hanging... need to be fixed")  # fixme
@pytest.mark.parametrize("flow", [StaticWebFrontendFlow, StreamlitFrontendFlow])
@mock.patch("lightning.app.runners.multiprocess.find_free_network_port")
def test_layout_leaf_node(find_ports_mock, flow):
    find_ports_mock.side_effect = lambda: 100
    flow = flow()
    app = LightningApp(flow)
    assert flow._layout == {}
    # we copy the dict here because after we dispatch the dict will get update with new instances
    # as the layout gets updated during the loop.
    frontends = app.frontends.copy()
    MultiProcessRuntime(app).dispatch()
    assert flow.counter == 3

    # The target url is available for the frontend after we started the servers in dispatch
    assert flow._layout == {"target": "http://localhost:100/root"}
    assert app.frontends[flow.name].flow is flow

    # we start the servers for the frontends that we collected at the time of app instantiation
    frontends[flow.name].start_server.assert_called_once()

    # leaf layout nodes can't be changed, they stay the same from when they first got configured
    assert app.frontends[flow.name] == frontends[flow.name]


def test_default_content_layout():
    class SimpleFlow(EmptyFlow):
        def configure_layout(self):
            frontend = StaticWebFrontend(serve_dir="a/b/c")
            frontend.start_server = Mock()
            return frontend

    class TestContentComponent(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.component0 = SimpleFlow()
            self.component1 = SimpleFlow()
            self.component2 = SimpleFlow()

    root = TestContentComponent()
    LightningApp(root)
    assert root._layout == [
        {"name": "root.component0", "content": "root.component0"},
        {"name": "root.component1", "content": "root.component1"},
        {"name": "root.component2", "content": "root.component2"},
    ]


def test_url_content_layout():
    class TestContentComponent(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.component0 = EmptyFlow()
            self.component1 = EmptyFlow()

        def configure_layout(self):
            return [
                {"name": "one", "content": self.component0},
                {"name": "url", "content": "https://lightning.ai"},
                {"name": "two", "content": self.component1},
            ]

    root = TestContentComponent()
    LightningApp(root)
    assert root._layout == [
        {"name": "one", "content": "root.component0"},
        {"name": "url", "content": "https://lightning.ai", "target": "https://lightning.ai"},
        {"name": "two", "content": "root.component1"},
    ]


def test_single_content_layout():
    """Test that returning a single dict also works (does not have to be returned in a list)."""

    class TestContentComponent(EmptyFlow):
        def __init__(self):
            super().__init__()
            self.component0 = EmptyFlow()
            self.component1 = EmptyFlow()

        def configure_layout(self):
            return {"name": "single", "content": self.component1}

    root = TestContentComponent()
    LightningApp(root)
    assert root._layout == [{"name": "single", "content": "root.component1"}]


class DynamicContentComponent(EmptyFlow):
    def __init__(self):
        super().__init__()
        self.component0 = EmptyFlow()
        self.component1 = EmptyFlow()
        self.counter = 0
        self.configure_layout_called = 0

    def run(self):
        self.run_assertion()
        self.counter += 1
        if self.counter == 3:
            self.stop()

    def configure_layout(self):
        self.configure_layout_called += 1
        tabs = [
            {"name": "one", "content": self.component0},
            {"name": f"{self.counter}", "content": self.component1},
        ]
        # reverse the order of the two tabs every time the counter is odd
        if self.counter % 2 != 0:
            tabs = tabs[::-1]
        return tabs

    def run_assertion(self):
        """Assert that the layout changes as the counter changes its value."""
        layout_even = [
            {"name": "one", "content": "root.component0"},
            {"name": f"{self.counter}", "content": "root.component1"},
        ]
        layout_odd = layout_even[::-1]
        assert (
            self.counter % 2 == 0
            and self._layout == layout_even
            or self.counter % 2 == 1
            and self._layout == layout_odd
        )


@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut exception")
@pytest.mark.xfail(strict=False, reason="hanging... need to be fixed")  # fixme
def test_dynamic_content_layout_update():
    """Test that the `configure_layout()` gets called as part of the loop and can return new layouts."""
    flow = DynamicContentComponent()
    app = LightningApp(flow)
    MultiProcessRuntime(app).dispatch()
    assert flow.configure_layout_called == 5


@mock.patch("lightning.app.utilities.layout.is_running_in_cloud", return_value=True)
def test_http_url_warning(*_):
    class Root(EmptyFlow):
        def configure_layout(self):
            return [
                {"name": "warning expected", "content": "http://github.com/very/long/link/to/display"},
                {"name": "no warning expected", "content": "https://github.com"},
            ]

    root = Root()

    with pytest.warns(
        UserWarning,
        match=escape("You configured an http link http://github.com/very/long/link... but it won't be accessible"),
    ):
        LightningApp(root)
