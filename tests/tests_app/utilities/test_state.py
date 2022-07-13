import os
from re import escape
from unittest import mock

import pytest
import requests

import lightning_app
from lightning_app import LightningApp, LightningFlow, LightningWork
from lightning_app.utilities.app_helpers import AppStatePlugin, BaseStatePlugin
from lightning_app.utilities.state import AppState


@mock.patch("lightning_app.utilities.state._configure_session", return_value=requests)
def test_app_state_not_connected(_):

    """Test an error message when a disconnected AppState tries to access attributes."""
    state = AppState()
    with pytest.raises(AttributeError, match="Failed to connect and fetch the app state"):
        _ = state.value
    with pytest.raises(AttributeError, match="Failed to connect and fetch the app state"):
        state.value = 1


@pytest.mark.parametrize(
    "my_affiliation,global_affiliation,expected",
    [
        (None, (), ()),
        ((), (), ()),
        ((), ("a", "b"), ()),
        (None, ("a", "b"), ("a", "b")),
    ],
)
@mock.patch("lightning_app.utilities.state._configure_session", return_value=requests)
def test_app_state_affiliation(_, my_affiliation, global_affiliation, expected):
    AppState._MY_AFFILIATION = global_affiliation
    state = AppState(my_affiliation=my_affiliation)
    assert state._my_affiliation == expected
    AppState._MY_AFFILIATION = ()


def test_app_state_state_access():
    """Test the many ways an AppState object can be accessed to set or get attributes on the state."""
    mocked_state = dict(
        vars=dict(root_var="root"),
        flows=dict(
            child0=dict(
                vars=dict(child_var=1),
                flows=dict(),
                works=dict(),
            )
        ),
        works=dict(
            work0=dict(
                vars=dict(work_var=2),
                flows=dict(),
                works=dict(),
            )
        ),
    )

    state = AppState()
    state._state = state._last_state = mocked_state

    assert state.root_var == "root"
    assert isinstance(state.child0, AppState)
    assert isinstance(state.work0, AppState)
    assert state.child0.child_var == 1
    assert state.work0.work_var == 2

    with pytest.raises(AttributeError, match="Failed to access 'non_existent_var' through `AppState`."):
        _ = state.work0.non_existent_var

    with pytest.raises(AttributeError, match="Failed to access 'non_existent_var' through `AppState`."):
        state.work0.non_existent_var = 22

    # TODO: improve msg
    with pytest.raises(AttributeError, match="You shouldn't set the flows"):
        state.child0 = "child0"

    # TODO: verify with tchaton
    with pytest.raises(AttributeError, match="You shouldn't set the works"):
        state.work0 = "work0"


@mock.patch("lightning_app.utilities.state.AppState.send_delta")
def test_app_state_state_access_under_affiliation(*_):
    """Test the access to attributes when the state is restricted under the given affiliation."""
    mocked_state = dict(
        vars=dict(root_var="root"),
        flows=dict(
            child0=dict(
                vars=dict(child0_var=0),
                flows=dict(
                    child1=dict(
                        vars=dict(child1_var=1),
                        flows=dict(
                            child2=dict(
                                vars=dict(child2_var=2),
                                flows=dict(),
                                works=dict(),
                            ),
                        ),
                        works=dict(),
                    ),
                ),
                works=dict(
                    work1=dict(
                        vars=dict(work1_var=11),
                    ),
                ),
            ),
        ),
        works=dict(),
    )

    # root-level affiliation
    state = AppState(my_affiliation=())
    state._store_state(mocked_state)
    assert isinstance(state.child0, AppState)
    assert state.child0.child0_var == 0
    assert state.child0.child1.child1_var == 1
    assert state.child0.child1.child2.child2_var == 2

    # one child deep
    state = AppState(my_affiliation=("child0",))
    state._store_state(mocked_state)
    assert state._state == mocked_state["flows"]["child0"]
    with pytest.raises(AttributeError, match="Failed to access 'child0' through `AppState`"):
        _ = state.child0
    assert state.child0_var == 0
    assert state.child1.child1_var == 1
    assert state.child1.child2.child2_var == 2

    # two flows deep
    state = AppState(my_affiliation=("child0", "child1"))
    state._store_state(mocked_state)
    assert state._state == mocked_state["flows"]["child0"]["flows"]["child1"]
    with pytest.raises(AttributeError, match="Failed to access 'child1' through `AppState`"):
        _ = state.child1
    state.child1_var = 111
    assert state.child1_var == 111
    assert state.child2.child2_var == 2

    # access to work
    state = AppState(my_affiliation=("child0", "work1"))
    state._store_state(mocked_state)
    assert state._state == mocked_state["flows"]["child0"]["works"]["work1"]
    with pytest.raises(AttributeError, match="Failed to access 'child1' through `AppState`"):
        _ = state.child1
    assert state.work1_var == 11
    state.work1_var = 111
    assert state.work1_var == 111

    # affiliation does not match state
    state = AppState(my_affiliation=("child1", "child0"))
    with pytest.raises(
        ValueError, match=escape("Failed to extract the state under the affiliation '('child1', 'child0')'")
    ):
        state._store_state(mocked_state)


def test_app_state_repr():
    app_state = AppState()
    assert repr(app_state) == "None"

    app_state = AppState()
    app_state._store_state(dict(vars=dict(x=1, y=2)))
    assert repr(app_state) == "{'vars': {'x': 1, 'y': 2}}"

    app_state = AppState()
    app_state._store_state(dict(vars=dict(x=1, y=2)))
    assert repr(app_state.y) == "2"

    app_state = AppState()
    app_state._store_state(dict(vars={}, flows=dict(child=dict(vars=dict(child_var="child_val")))))
    assert repr(app_state.child) == "{'vars': {'child_var': 'child_val'}}"


def test_app_state_bool():
    app_state = AppState()
    assert not bool(app_state)

    app_state = AppState()
    app_state._store_state(dict(vars=dict(x=1, y=2)))
    assert bool(app_state)


class _CustomAppStatePlugin(BaseStatePlugin):
    def should_update_app(self, deep_diff):
        pass

    def get_context(self):
        pass

    def render_non_authorized(self):
        pass


def test_attach_plugin():
    """Test how plugins get attached to the AppState and the default behavior when no plugin is specified."""
    app_state = AppState()
    assert isinstance(app_state._plugin, AppStatePlugin)

    app_state = AppState(plugin=_CustomAppStatePlugin())
    assert isinstance(app_state._plugin, _CustomAppStatePlugin)


@mock.patch("lightning_app.utilities.state._configure_session", return_value=requests)
def test_app_state_connection_error(_):
    """Test an error message when a connection to retrieve the state can't be established."""
    app_state = AppState()
    with pytest.raises(AttributeError, match=r"Failed to connect and fetch the app state\. Is the app running?"):
        app_state._request_state()

    with pytest.raises(AttributeError, match=r"Failed to connect and fetch the app state\. Is the app running?"):
        app_state.var = 1


class Work(LightningWork):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def run(self):
        self.counter += 1


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.should_start = False
        self.w = Work()

    def run(self):
        if self.should_start:
            self.w.run()
            self._exit()


class MockResponse:
    def __init__(self, state, status_code):
        self._state = state
        self.status_code = status_code

    def json(self):
        return self._state


def test_get_send_request(monkeypatch):

    app = LightningApp(Flow())
    monkeypatch.setattr(lightning_app.utilities.state, "_configure_session", mock.MagicMock())

    state = AppState(plugin=AppStatePlugin())
    state._session.get._mock_return_value = MockResponse(app.state_with_changes, 500)
    state._request_state()
    state._session.get._mock_return_value = MockResponse(app.state_with_changes, 200)
    state._request_state()
    assert state._my_affiliation == ()
    with pytest.raises(Exception, match="The response from"):
        state._session.post._mock_return_value = MockResponse(app.state_with_changes, 500)
        state.w.counter = 1
    state._session.post._mock_return_value = MockResponse(app.state_with_changes, 200)
    state.w.counter = 1


@mock.patch("lightning_app.utilities.state.APP_SERVER_HOST", "https://lightning-cloud.com")
@mock.patch.dict(os.environ, {"LIGHTNING_APP_STATE_URL": "https://lightning-cloud.com"})
def test_app_state_with_env_var(**__):
    state = AppState()
    assert state._host == "https://lightning-cloud.com"
    assert not state._port
    assert state._url == "https://lightning-cloud.com"


@mock.patch.dict(os.environ, {})
def test_app_state_with_no_env_var(**__):
    state = AppState()
    assert state._host == "http://127.0.0.1"
    assert state._port == 7501
    assert state._url == "http://127.0.0.1:7501"
