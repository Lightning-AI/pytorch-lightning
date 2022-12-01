import pytest

from lightning_app.core.flow import LightningFlow
from lightning_app.core.work import LightningWork
from lightning_app.frontend.web import StaticWebFrontend
from lightning_app.utilities.layout import _collect_layout


class _MockApp:
    def __init__(self) -> None:
        self.frontends = {}


class FlowWithFrontend(LightningFlow):
    def configure_layout(self):
        return StaticWebFrontend(".")


class WorkWithFrontend(LightningWork):
    def run(self):
        pass

    def configure_layout(self):
        return StaticWebFrontend(".")


class FlowWithWorkWithFrontend(LightningFlow):
    def __init__(self):
        super().__init__()

        self.work = WorkWithFrontend()

    def configure_layout(self):
        return {"name": "work", "content": self.work}


class FlowWithUrl(LightningFlow):
    def configure_layout(self):
        return {"name": "test", "content": "https://test"}


class WorkWithUrl(LightningWork):
    def run(self):
        pass

    def configure_layout(self):
        return "https://test"


class FlowWithWorkWithUrl(LightningFlow):
    def __init__(self):
        super().__init__()

        self.work = WorkWithUrl()

    def configure_layout(self):
        return {"name": "test", "content": self.work}


@pytest.mark.parametrize(
    "flow,expected_layout,expected_frontends",
    [
        (FlowWithFrontend, {}, [("root", StaticWebFrontend)]),
        (FlowWithWorkWithFrontend, {}, [("root", StaticWebFrontend)]),
        (FlowWithUrl, [{"name": "test", "content": "https://test", "target": "https://test"}], []),
        (FlowWithWorkWithUrl, [{"name": "test", "content": "https://test", "target": "https://test"}], []),
    ],
)
def test_collect_layout(flow, expected_layout, expected_frontends):
    app = _MockApp()
    flow = flow()
    layout = _collect_layout(app, flow)

    assert layout == expected_layout
    assert set(app.frontends.keys()) == {key for key, _ in expected_frontends}
    for key, frontend_type in expected_frontends:
        assert isinstance(app.frontends[key], frontend_type)
