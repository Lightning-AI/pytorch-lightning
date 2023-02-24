# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import json
import os
from copy import deepcopy
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

from deepdiff import DeepDiff
from requests import Session
from requests.exceptions import ConnectionError

from lightning.app.core.constants import APP_SERVER_HOST, APP_SERVER_PORT
from lightning.app.storage.drive import _maybe_create_drive
from lightning.app.utilities.app_helpers import AppStatePlugin, BaseStatePlugin, Logger
from lightning.app.utilities.network import _configure_session

logger = Logger(__name__)

# GLOBAL APP STATE
_LAST_STATE = None
_STATE = None


class AppStateType(enum.Enum):
    STREAMLIT = enum.auto()
    DEFAULT = enum.auto()


def headers_for(context: Dict[str, str]) -> Dict[str, str]:
    return {
        "X-Lightning-Session-UUID": context.get("token", ""),
        "X-Lightning-Session-ID": context.get("session_id", ""),
        "X-Lightning-Type": context.get("type", ""),
    }


class AppState:
    _APP_PRIVATE_KEYS: Tuple[str, ...] = (
        "_host",
        "_session_id",
        "_state",
        "_last_state",
        "_url",
        "_port",
        "_request_state",
        "_store_state",
        "_send_state",
        "_my_affiliation",
        "_find_state_under_affiliation",
        "_plugin",
        "_attach_plugin",
        "_authorized",
        "is_authorized",
        "_debug",
        "_session",
    )
    _MY_AFFILIATION: Tuple[str, ...] = ()

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        last_state: Optional[Dict] = None,
        state: Optional[Dict] = None,
        my_affiliation: Tuple[str, ...] = None,
        plugin: Optional[BaseStatePlugin] = None,
    ) -> None:
        """The AppState class enables Frontend users to interact with their application state.

        When the state isn't defined, it would be pulled from the app REST API Server.
        If the state gets modified by the user, the new state would be sent to the API Server.

        Arguments:
            host: Rest API Server current host
            port: Rest API Server current port
            last_state: The state pulled on first access.
            state: The state modified by the user.
            my_affiliation: A tuple describing the affiliation this app state represents. When storing a state dict
                on this AppState, this affiliation will be used to reduce the scope of the given state.
            plugin: A plugin to handle authorization.
        """
        use_localhost = "LIGHTNING_APP_STATE_URL" not in os.environ
        self._host = host or APP_SERVER_HOST
        self._port = port or (APP_SERVER_PORT if use_localhost else None)
        self._url = f"{self._host}:{self._port}" if use_localhost else self._host
        self._last_state = last_state
        self._state = state
        self._session_id = "1234"
        self._my_affiliation = my_affiliation if my_affiliation is not None else AppState._MY_AFFILIATION
        self._authorized = None
        self._attach_plugin(plugin)
        self._session = self._configure_session()

    def _attach_plugin(self, plugin: Optional[BaseStatePlugin]) -> None:
        if plugin is not None:
            plugin = plugin
        else:
            plugin = AppStatePlugin()
        self._plugin = plugin

    @staticmethod
    def _find_state_under_affiliation(state, my_affiliation: Tuple[str, ...]) -> Dict[str, Any]:
        """This method is used to extract the subset of the app state associated with the given affiliation.

        For example, if the affiliation is ``("root", "subflow")``, then the returned state will be
        ``state["flows"]["subflow"]``.
        """
        children_state = state
        for name in my_affiliation:
            if name in children_state["flows"]:
                children_state = children_state["flows"][name]
            elif name in children_state["works"]:
                children_state = children_state["works"][name]
            else:
                raise ValueError(f"Failed to extract the state under the affiliation '{my_affiliation}'.")
        return children_state

    def _store_state(self, state: Dict[str, Any]) -> None:
        # Relying on the global variable to ensure the
        # deep_diff is done on the entire state.
        global _LAST_STATE
        global _STATE
        _LAST_STATE = deepcopy(state)
        _STATE = state
        # If the affiliation is passed, the AppState was created in a LightningFlow context.
        # The state should be only the one of this LightningFlow and its children.
        self._last_state = self._find_state_under_affiliation(_LAST_STATE, self._my_affiliation)
        self._state = self._find_state_under_affiliation(_STATE, self._my_affiliation)

    def send_delta(self) -> None:
        app_url = f"{self._url}/api/v1/delta"
        deep_diff = DeepDiff(_LAST_STATE, _STATE, verbose_level=2)
        assert self._plugin is not None
        # TODO: Find how to prevent the infinite loop on refresh without storing the DeepDiff
        if self._plugin.should_update_app(deep_diff):
            data = {"delta": json.loads(deep_diff.to_json())}
            headers = headers_for(self._plugin.get_context())
            try:
                # TODO: Send the delta directly to the REST API.
                response = self._session.post(app_url, json=data, headers=headers)
            except ConnectionError as e:
                raise AttributeError("Failed to connect and send the app state. Is the app running?") from e

            if response and response.status_code != 200:
                raise Exception(f"The response from the server was {response.status_code}. Your inputs were rejected.")

    def _request_state(self) -> None:
        if self._state is not None:
            return
        app_url = f"{self._url}/api/v1/state"
        headers = headers_for(self._plugin.get_context()) if self._plugin else {}

        response_json = {}

        # Sometimes the state URL can return an empty JSON when things are being set-up,
        # so we wait for it to be ready here.
        while response_json == {}:
            sleep(0.5)
            try:
                response = self._session.get(app_url, headers=headers, timeout=1)
            except ConnectionError as e:
                raise AttributeError("Failed to connect and fetch the app state. Is the app running?") from e

            self._authorized = response.status_code
            if self._authorized != 200:
                return

            response_json = response.json()

        logger.debug(f"GET STATE {response} {response_json}")
        self._store_state(response_json)

    def __getattr__(self, name: str) -> Union[Any, "AppState"]:
        if name in self._APP_PRIVATE_KEYS:
            return object.__getattr__(self, name)

        # The state needs to be fetched on access if it doesn't exist.
        self._request_state()

        if name in self._state.get("vars", {}):
            value = self._state["vars"][name]
            if isinstance(value, dict):
                return _maybe_create_drive("root." + ".".join(self._my_affiliation), value)
            return value

        elif name in self._state.get("works", {}):
            return AppState(
                self._host, self._port, last_state=self._last_state["works"][name], state=self._state["works"][name]
            )

        elif name in self._state.get("flows", {}):
            return AppState(
                self._host,
                self._port,
                last_state=self._last_state["flows"][name],
                state=self._state["flows"][name],
            )

        elif name in self._state.get("structures", {}):
            return AppState(
                self._host,
                self._port,
                last_state=self._last_state["structures"][name],
                state=self._state["structures"][name],
            )

        raise AttributeError(
            f"Failed to access '{name}' through `AppState`. The state provides:"
            f" Variables: {list(self._state['vars'].keys())},"
            f" Components: {list(self._state.get('flows', {}).keys()) + list(self._state.get('works', {}).keys())}",
        )

    def __getitem__(self, key: str):
        return self.__getattr__(key)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._APP_PRIVATE_KEYS:
            object.__setattr__(self, name, value)
            return

        # The state needs to be fetched on access if it doesn't exist.
        self._request_state()

        # TODO: Find a way to aggregate deltas to avoid making
        # request for each attribute change.
        if name in self._state["vars"]:
            self._state["vars"][name] = value
            self.send_delta()

        elif name in self._state["flows"]:
            raise AttributeError("You shouldn't set the flows directly onto the state. Use its attributes instead.")

        elif name in self._state["works"]:
            raise AttributeError("You shouldn't set the works directly onto the state. Use its attributes instead.")

        else:
            raise AttributeError(
                f"Failed to access '{name}' through `AppState`. The state provides:"
                f" Variables: {list(self._state['vars'].keys())},"
                f" Components: {list(self._state['flows'].keys()) + list(self._state['works'].keys())}",
            )

    def __repr__(self) -> str:
        return str(self._state)

    def __bool__(self) -> bool:
        return bool(self._state)

    def __len__(self) -> int:
        # The state needs to be fetched on access if it doesn't exist.
        self._request_state()

        keys = []
        for component in ["flows", "works", "structures"]:
            keys.extend(list(self._state.get(component, {})))
        return len(keys)

    def items(self) -> List[Dict[str, Any]]:
        # The state needs to be fetched on access if it doesn't exist.
        self._request_state()

        items = []
        for component in ["flows", "works"]:
            state = self._state.get(component, {})
            last_state = self._last_state.get(component, {})
            for name, state_value in state.items():
                v = AppState(
                    self._host,
                    self._port,
                    last_state=last_state[name],
                    state=state_value,
                )
                items.append((name, v))

        structures = self._state.get("structures", {})
        last_structures = self._last_state.get("structures", {})
        if structures:
            for component in ["flows", "works"]:
                state = structures.get(component, {})
                last_state = last_structures.get(component, {})
                for name, state_value in state.items():
                    v = AppState(
                        self._host,
                        self._port,
                        last_state=last_state[name],
                        state=state_value,
                    )
                    items.append((name, v))
        return items

    @staticmethod
    def _configure_session() -> Session:
        return _configure_session()
