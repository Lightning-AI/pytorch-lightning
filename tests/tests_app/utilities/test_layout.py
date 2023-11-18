import pytest
from lightning.app.core.flow import LightningFlow
from lightning.app.core.work import LightningWork
from lightning.app.frontend.web import StaticWebFrontend
from lightning.app.utilities.layout import _collect_layout


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
    ("flow", "expected_layout", "expected_frontends"),
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


class FlowWithBadLayout(LightningFlow):
    def configure_layout(self):
        return 100


class FlowWithBadLayoutDict(LightningFlow):
    def configure_layout(self):
        return {"this_key_should_not_be_here": "http://appurl"}


class FlowWithBadContent(LightningFlow):
    def configure_layout(self):
        return {"content": 100}


class WorkWithBadLayout(LightningWork):
    def run(self):
        pass

    def configure_layout(self):
        return 100


class FlowWithWorkWithBadLayout(LightningFlow):
    def __init__(self):
        super().__init__()

        self.work = WorkWithBadLayout()

    def configure_layout(self):
        return {"name": "test", "content": self.work}


class FlowWithMultipleWorksWithFrontends(LightningFlow):
    def __init__(self):
        super().__init__()

        self.work1 = WorkWithFrontend()
        self.work2 = WorkWithFrontend()

    def configure_layout(self):
        return [{"name": "test1", "content": self.work1}, {"name": "test2", "content": self.work2}]


@pytest.mark.parametrize(
    ("flow", "error_type", "match"),
    [
        (FlowWithBadLayout, TypeError, "is an unsupported layout format"),
        (FlowWithBadLayoutDict, ValueError, "missing a key 'content'."),
        (FlowWithBadContent, ValueError, "contains an unsupported entry."),
        (FlowWithWorkWithBadLayout, TypeError, "is of an unsupported type."),
        (
            FlowWithMultipleWorksWithFrontends,
            TypeError,
            "The tab containing a `WorkWithFrontend` must be the only tab",
        ),
    ],
)
def test_collect_layout_errors(flow, error_type, match):
    app = _MockApp()
    flow = flow()

    with pytest.raises(error_type, match=match):
        _collect_layout(app, flow)
