import os
from re import escape
from unittest import mock

import lightning.app
import pytest
import requests
from lightning.app import LightningApp, LightningFlow, LightningWork
from lightning.app.structures import Dict, List
from lightning.app.utilities.app_helpers import AppStatePlugin, BaseStatePlugin
from lightning.app.utilities.state import AppState
from lightning_cloud.openapi import Externalv1LightningappInstance, V1LightningappInstanceStatus


@mock.patch("lightning.app.utilities.state._configure_session", return_value=requests)
def test_app_state_not_connected(_):
    """Test an error message when a disconnected AppState tries to access attributes."""
    state = AppState(port=8000)
    with pytest.raises(AttributeError, match="Failed to connect and fetch the app state"):
        _ = state.value
    with pytest.raises(AttributeError, match="Failed to connect and fetch the app state"):
        state.value = 1


@pytest.mark.parametrize(
    ("my_affiliation", "global_affiliation", "expected"),
    [
        (None, (), ()),
        ((), (), ()),
        ((), ("a", "b"), ()),
        (None, ("a", "b"), ("a", "b")),
    ],
)
@mock.patch("lightning.app.utilities.state._configure_session", return_value=requests)
def test_app_state_affiliation(_, my_affiliation, global_affiliation, expected):
    AppState._MY_AFFILIATION = global_affiliation
    state = AppState(my_affiliation=my_affiliation)
    assert state._my_affiliation == expected
    AppState._MY_AFFILIATION = ()


def test_app_state_state_access():
    """Test the many ways an AppState object can be accessed to set or get attributes on the state."""
    mocked_state = {
        "vars": {"root_var": "root"},
        "flows": {
            "child0": {
                "vars": {"child_var": 1},
                "flows": {},
                "works": {},
            }
        },
        "works": {
            "work0": {
                "vars": {"work_var": 2},
                "flows": {},
                "works": {},
            }
        },
    }

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


@mock.patch("lightning.app.utilities.state.AppState.send_delta")
def test_app_state_state_access_under_affiliation(*_):
    """Test the access to attributes when the state is restricted under the given affiliation."""
    mocked_state = {
        "vars": {"root_var": "root"},
        "flows": {
            "child0": {
                "vars": {"child0_var": 0},
                "flows": {
                    "child1": {
                        "vars": {"child1_var": 1},
                        "flows": {
                            "child2": {
                                "vars": {"child2_var": 2},
                                "flows": {},
                                "works": {},
                            },
                        },
                        "works": {},
                    },
                },
                "works": {
                    "work1": {
                        "vars": {"work1_var": 11},
                    },
                },
            },
        },
        "works": {},
    }

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
    app_state._store_state({"vars": {"x": 1, "y": 2}})
    assert repr(app_state) == "{'vars': {'x': 1, 'y': 2}}"

    app_state = AppState()
    app_state._store_state({"vars": {"x": 1, "y": 2}})
    assert repr(app_state.y) == "2"

    app_state = AppState()
    app_state._store_state({"vars": {}, "flows": {"child": {"vars": {"child_var": "child_val"}}}})
    assert repr(app_state.child) == "{'vars': {'child_var': 'child_val'}}"


def test_app_state_bool():
    app_state = AppState()
    assert not bool(app_state)

    app_state = AppState()
    app_state._store_state({"vars": {"x": 1, "y": 2}})
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


@mock.patch("lightning.app.utilities.state._configure_session", return_value=requests)
def test_app_state_connection_error(_):
    """Test an error message when a connection to retrieve the state can't be established."""
    app_state = AppState(port=8000)
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
            self.stop()


class MockResponse:
    def __init__(self, state, status_code):
        self._state = state
        self.status_code = status_code

    def json(self):
        return self._state


def test_get_send_request(monkeypatch):
    app = LightningApp(Flow())
    monkeypatch.setattr(lightning.app.utilities.state, "_configure_session", mock.MagicMock())

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


@mock.patch.dict(
    os.environ,
    {
        "LIGHTNING_APP_STATE_URL": "https://lightning-cloud.com",
        "LIGHTNING_CLOUD_PROJECT_ID": "test-project-id",
        "LIGHTNING_CLOUD_APP_ID": "test-app-id",
    },
)
@mock.patch("lightning.app.utilities.state.LightningClient")
def test_app_state_with_env_var(mock_client):
    mock_client().lightningapp_instance_service_get_lightningapp_instance.return_value = Externalv1LightningappInstance(
        status=V1LightningappInstanceStatus(ip_address="test-ip"),
    )
    state = AppState()
    url = state._url

    mock_client().lightningapp_instance_service_get_lightningapp_instance.assert_called_once_with(
        "test-project-id",
        "test-app-id",
    )

    assert url == "http://test-ip:8080"
    assert not state._port


@mock.patch.dict(os.environ, {})
def test_app_state_with_no_env_var(**__):
    state = AppState()
    assert state._host == "http://127.0.0.1"
    assert state._port == 7501
    assert state._url == "http://127.0.0.1:7501"


class FlowStructures(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w_list = List(Work(), Work())
        self.w_dict = Dict(**{"toto": Work(), "toto_2": Work()})

    def run(self):
        self.stop()


class FlowStructuresEmpty(LightningFlow):
    def __init__(self):
        super().__init__()
        self.w_list = List()
        self.w_dict = Dict()

    def run(self):
        self.stop()


def test_app_state_with_structures():
    app = LightningApp(FlowStructures())
    state = AppState()
    state._last_state = app.state
    state._state = app.state
    assert state.w_list["0"].counter == 0
    assert len(state.w_list) == 2
    assert state.w_dict["toto"].counter == 0
    assert [k for k, _ in state.w_dict.items()] == ["toto", "toto_2"]
    assert [k for k, _ in state.w_list.items()] == ["0", "1"]

    app = LightningApp(FlowStructuresEmpty())
    state = AppState()
    state._last_state = app.state
    state._state = app.state
    assert state.w_list
