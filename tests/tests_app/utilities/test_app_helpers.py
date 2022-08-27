from unittest import mock
from unittest.mock import Mock

import pytest

from lightning_app import LightningFlow, LightningWork
from lightning_app.utilities.app_helpers import (
    AppStatePlugin,
    BaseStatePlugin,
    InMemoryStateStore,
    is_overridden,
    StateStore,
)
from lightning_app.utilities.exceptions import LightningAppStateException


class Work(LightningWork):
    def run(self):
        pass


class Flow(LightningFlow):
    def run(self):
        pass


def test_is_overridden():
    flow = Flow()
    work = Work()

    # edge cases
    assert not is_overridden("whatever", None)
    with pytest.raises(ValueError, match="Expected a parent"):
        is_overridden("whatever", object())
    assert not is_overridden("whatever", flow)
    assert not is_overridden("whatever", flow, parent=Flow)

    class TestFlow(LightningFlow):
        def run(self):
            pass

        def foo(self):
            pass

        def bar(self):
            return 1

    with pytest.raises(ValueError, match="The parent should define the method"):
        is_overridden("foo", TestFlow())

    # normal usage
    assert is_overridden("run", flow)
    assert is_overridden("run", work)

    # `Mock` support
    mock = Mock(spec=Flow, wraps=flow)
    assert is_overridden("run", mock)
    mock = Mock(spec=LightningWork, wraps=work)
    assert is_overridden("run", mock)


def test_simple_app_store():

    store = InMemoryStateStore()
    user_id = "1234"
    store.add(user_id)
    state = {"data": user_id}
    store.set_app_state(user_id, state)
    store.set_served_state(user_id, state)
    store.set_served_session_id(user_id, user_id)
    assert store.get_app_state(user_id) == state
    assert store.get_served_state(user_id) == state
    assert store.get_served_session_id(user_id) == user_id
    store.remove(user_id)
    assert isinstance(store, StateStore)


@mock.patch("lightning_app.utilities.app_helpers.APP_STATE_MAX_SIZE_BYTES", 120)
def test_simple_app_store_warning():
    store = InMemoryStateStore()
    user_id = "1234"
    store.add(user_id)
    state = {"data": "I'm a state that's larger than 120 bytes"}
    with pytest.raises(LightningAppStateException, match="is larger than the"):
        store.set_app_state(user_id, state)


def test_base_state_plugin():
    class DummyStatePlugin(BaseStatePlugin):
        def should_update_app(self, deep_diff):
            super().should_update_app(deep_diff)

        def get_context(self):
            super().get_context()

        def render_non_authorized(self):
            super().render_non_authorized()

    plugin = DummyStatePlugin()
    plugin.should_update_app(None)
    plugin.get_context()
    plugin.render_non_authorized()

    plugin = AppStatePlugin()
    plugin.should_update_app(None)
    plugin.get_context()
    plugin.render_non_authorized()
