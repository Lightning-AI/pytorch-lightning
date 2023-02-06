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

import abc
import asyncio
import builtins
import enum
import functools
import inspect
import json
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple, Type, TYPE_CHECKING
from unittest.mock import MagicMock

import websockets
from deepdiff import Delta
from lightning_cloud.openapi import AppinstancesIdBody, Externalv1LightningappInstance, V1LightningappInstanceState

import lightning_app
from lightning_app.utilities.exceptions import LightningAppStateException
from lightning_app.utilities.tree import breadth_first

if TYPE_CHECKING:
    from lightning_app.core.app import LightningApp
    from lightning_app.core.flow import LightningFlow
    from lightning_app.utilities.types import Component

logger = logging.getLogger(__name__)


@dataclass
class StateEntry:
    """dataclass used to keep track the latest state shared through the app REST API."""

    app_state: Mapping = field(default_factory=dict)
    served_state: Mapping = field(default_factory=dict)
    session_id: Optional[str] = None


class StateStore(ABC):
    """Base class of State store that provides simple key, value store to keep track of app state, served app
    state."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add(self, k: str):
        """Creates a new empty state with input key 'k'."""
        pass

    @abstractmethod
    def remove(self, k: str):
        """Deletes a state with input key 'k'."""
        pass

    @abstractmethod
    def get_app_state(self, k: str) -> Mapping:
        """returns a stored appstate for an input key 'k'."""
        pass

    @abstractmethod
    def get_served_state(self, k: str) -> Mapping:
        """returns a last served app state for an input key 'k'."""
        pass

    @abstractmethod
    def get_served_session_id(self, k: str) -> str:
        """returns session id for state of a key 'k'."""
        pass

    @abstractmethod
    def set_app_state(self, k: str, v: Mapping):
        """sets the app state for state of a key 'k'."""
        pass

    @abstractmethod
    def set_served_state(self, k: str, v: Mapping):
        """sets the served state for state of a key 'k'."""
        pass

    @abstractmethod
    def set_served_session_id(self, k: str, v: str):
        """sets the session id for state of a key 'k'."""
        pass


class InMemoryStateStore(StateStore):
    """In memory simple store to keep track of state through the app REST API."""

    def __init__(self):
        self.store = {}
        self.counter = 0

    def add(self, k):
        self.store[k] = StateEntry()

    def remove(self, k):
        del self.store[k]

    def get_app_state(self, k):
        return self.store[k].app_state

    def get_served_state(self, k):
        return self.store[k].served_state

    def get_served_session_id(self, k):
        return self.store[k].session_id

    def set_app_state(self, k, v):
        state_size = sys.getsizeof(v)
        if state_size > lightning_app.core.constants.APP_STATE_MAX_SIZE_BYTES:
            raise LightningAppStateException(
                f"App state size is {state_size} bytes, which is larger than the recommended size "
                f"of {lightning_app.core.constants.APP_STATE_MAX_SIZE_BYTES}. Please investigate this."
            )
        self.store[k].app_state = deepcopy(v)
        self.counter += 1

    def set_served_state(self, k, v):
        self.store[k].served_state = deepcopy(v)

    def set_served_session_id(self, k, v):
        self.store[k].session_id = v


class _LightningAppRef:
    _app_instance: Optional["LightningApp"] = None

    @classmethod
    def connect(cls, app_instance: "LightningApp") -> None:
        cls._app_instance = app_instance

    @classmethod
    def get_current(cls) -> Optional["LightningApp"]:
        if cls._app_instance:
            return cls._app_instance


def affiliation(component: "Component") -> Tuple[str, ...]:
    """Returns the affiliation of a component."""
    if component.name in ("root", ""):
        return ()
    return tuple(component.name.split(".")[1:])


class AppStateType(str, enum.Enum):
    STREAMLIT = "STREAMLIT"
    DEFAULT = "DEFAULT"


class BaseStatePlugin(abc.ABC):
    def __init__(self):
        self.authorized = None

    @abc.abstractmethod
    def should_update_app(self, deep_diff):
        pass

    @abc.abstractmethod
    def get_context(self):
        pass

    @abc.abstractmethod
    def render_non_authorized(self):
        pass


class AppStatePlugin(BaseStatePlugin):
    def should_update_app(self, deep_diff):
        return deep_diff

    def get_context(self):
        return {"type": AppStateType.DEFAULT.value}

    def render_non_authorized(self):
        pass


def target_fn():
    try:
        # streamlit >= 1.14.0
        from streamlit import runtime

        get_instance = runtime.get_instance
        exists = runtime.exists()
    except ImportError:
        # Older versions
        from streamlit.server.server import Server

        get_instance = Server.get_current
        exists = bool(Server._singleton)

    async def update_fn():
        runtime_instance = get_instance()
        sessions = list(runtime_instance._session_info_by_id.values())
        url = (
            "localhost:8080"
            if "LIGHTNING_APP_STATE_URL" in os.environ
            else f"localhost:{lightning_app.core.constants.APP_SERVER_PORT}"
        )
        ws_url = f"ws://{url}/api/v1/ws"
        last_updated = time.time()
        async with websockets.connect(ws_url) as websocket:
            while True:
                try:
                    _ = await websocket.recv()

                    while (time.time() - last_updated) < 1:
                        time.sleep(0.1)
                    for session in sessions:
                        session = session.session
                        session.request_rerun(session._client_state)
                    last_updated = time.time()
                except websockets.exceptions.ConnectionClosedOK:
                    # The websocket is not enabled
                    break

    if exists:
        asyncio.run(update_fn())


class StreamLitStatePlugin(BaseStatePlugin):
    def __init__(self):
        super().__init__()
        import streamlit as st

        if hasattr(st, "session_state") and "websocket_thread" not in st.session_state:
            thread = threading.Thread(target=target_fn)
            st.session_state.websocket_thread = thread
            thread.setDaemon(True)
            thread.start()

    def should_update_app(self, deep_diff):
        return deep_diff

    def get_context(self):
        return {"type": AppStateType.DEFAULT.value}

    def render_non_authorized(self):
        pass


def is_overridden(method_name: str, instance: Optional[object] = None, parent: Optional[Type[object]] = None) -> bool:
    if instance is None:
        return False
    if parent is None:
        if isinstance(instance, lightning_app.LightningFlow):
            parent = lightning_app.LightningFlow
        elif isinstance(instance, lightning_app.LightningWork):
            parent = lightning_app.LightningWork
        if parent is None:
            raise ValueError("Expected a parent")
    from lightning_utilities.core.overrides import is_overridden

    return is_overridden(method_name, instance, parent)


def _is_json_serializable(x: Any) -> bool:
    """Test whether a variable can be encoded as json."""
    if type(x) in lightning_app.core.constants.SUPPORTED_PRIMITIVE_TYPES:
        # shortcut for primitive types that are not containers
        return True
    try:
        json.dumps(x, cls=LightningJSONEncoder)
        return True
    except (TypeError, OverflowError):
        # OverflowError is raised if number is too large to encode
        return False


def _set_child_name(component: "Component", child: "Component", new_name: str) -> str:
    """Computes and sets the name of a child given the parent, and returns the name."""
    child_name = f"{component.name}.{new_name}"
    child._name = child_name

    # the name changed, so recursively update the names of the children of this child
    if isinstance(child, lightning_app.core.LightningFlow):
        for n in child._flows:
            c = getattr(child, n)
            _set_child_name(child, c, n)
        for n in child._works:
            c = getattr(child, n)
            _set_child_name(child, c, n)
        for n in child._structures:
            s = getattr(child, n)
            _set_child_name(child, s, n)
    if isinstance(child, lightning_app.structures.Dict):
        for n, c in child.items():
            _set_child_name(child, c, n)
    if isinstance(child, lightning_app.structures.List):
        for c in child:
            _set_child_name(child, c, c.name.split(".")[-1])

    return child_name


def _delta_to_app_state_delta(root: "LightningFlow", component: "Component", delta: Delta) -> Delta:
    delta_dict = delta.to_dict()
    for changed in delta_dict.values():
        for delta_key in changed.copy().keys():
            val = changed[delta_key]

            new_prefix = "root"
            for p, c in _walk_to_component(root, component):

                if isinstance(c, lightning_app.core.LightningWork):
                    new_prefix += "['works']"

                if isinstance(c, lightning_app.core.LightningFlow):
                    new_prefix += "['flows']"

                if isinstance(c, (lightning_app.structures.Dict, lightning_app.structures.List)):
                    new_prefix += "['structures']"

                c_n = c.name.split(".")[-1]
                new_prefix += f"['{c_n}']"

            delta_key_without_root = delta_key[4:]  # the first 4 chars are the word 'root', strip it
            new_key = new_prefix + delta_key_without_root
            if new_key != delta_key:
                changed[new_key] = val
                del changed[delta_key]

    return Delta(delta_dict)


def _walk_to_component(
    root: "LightningFlow",
    component: "Component",
) -> Generator[Tuple["Component", "Component"], None, None]:
    """Returns a generator that runs through the tree starting from the root down to the given component.

    At each node, yields parent and child as a tuple.
    """
    from lightning_app.structures import Dict, List

    name_parts = component.name.split(".")[1:]  # exclude 'root' from the name
    parent = root
    for n in name_parts:
        if isinstance(parent, (Dict, List)):
            child = parent[n] if isinstance(parent, Dict) else parent[int(n)]
        else:
            child = getattr(parent, n)
        yield parent, child
        parent = child


def _collect_child_process_pids(pid: int) -> List[int]:
    """Function to return the list of child process pid's of a process."""
    processes = os.popen("ps -ej | grep -i 'python' | grep -v 'grep' | awk '{ print $2,$3 }'").read()
    processes = [p.split(" ") for p in processes.split("\n")[:-1]]
    return [int(child) for child, parent in processes if parent == str(pid) and child != str(pid)]


def _print_to_logger_info(*args, **kwargs):
    # TODO Find a better way to re-direct print to loggers.
    lightning_app._logger.info(" ".join([str(v) for v in args]))


def convert_print_to_logger_info(func: Callable) -> Callable:
    """This function is used to transform any print into logger.info calls, so it gets tracked in the cloud."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        original_print = __builtins__["print"]
        __builtins__["print"] = _print_to_logger_info
        res = func(*args, **kwargs)
        __builtins__["print"] = original_print
        return res

    return wrapper


def pretty_state(state: Dict) -> Dict:
    """Utility to prettify the state by removing hidden attributes."""
    new_state = {}
    for k, v in state["vars"].items():
        if not k.startswith("_"):
            if "vars" not in new_state:
                new_state["vars"] = {}
            new_state["vars"][k] = v
    if "flows" in state:
        for k, v in state["flows"].items():
            if "flows" not in new_state:
                new_state["flows"] = {}
            new_state["flows"][k] = pretty_state(state["flows"][k])
    if "works" in state:
        for k, v in state["works"].items():
            if "works" not in new_state:
                new_state["works"] = {}
            new_state["works"][k] = pretty_state(state["works"][k])
    return new_state


class LightningJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if callable(getattr(obj, "__json__", None)):
            return obj.__json__()
        return json.JSONEncoder.default(self, obj)


class Logger:
    """This class is used to improve the debugging experience."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.level = None

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._set_level()
        self.logger.warn(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._set_level()
        self.logger.debug(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._set_level()
        self.logger.error(msg, *args, **kwargs)

    def _set_level(self):
        """Lazily set the level once set by the users."""
        # Set on the first from either log, warn, debug or error call.
        if self.level is None:
            self.level = logging.DEBUG if bool(int(os.getenv("LIGHTNING_DEBUG", "0"))) else logging.INFO
            self.logger.setLevel(self.level)


def _state_dict(flow: "LightningFlow"):
    state = {}
    flows = [flow] + list(flow.flows.values())
    for f in flows:
        state[f.name] = f.state_dict()
    for w in flow.works():
        state[w.name] = w.state
    return state


def _load_state_dict(root_flow: "LightningFlow", state: Dict[str, Any], strict: bool = True) -> None:
    """This function is used to reload the state assuming dynamic components creation.

    When a component isn't found but its state exists, its state is passed up to its closest existing parent.

    Arguments:
        root_flow: The flow at the top of the component tree.
        state: The collected state dict.
        strict: Whether to validate all components have been re-created.
    """
    # 1: Reload the state of the existing works
    for w in root_flow.works():
        w.set_state(state.pop(w.name))

    # 2: Collect the existing flows
    flows = [root_flow] + list(root_flow.flows.values())
    flow_map = {f.name: f for f in flows}

    # 3: Find the state of the all dynamic components
    dynamic_components = {k: v for k, v in state.items() if k not in flow_map}

    # 4: Propagate the state of the dynamic components to their closest parents
    dynamic_children_state = {}
    for name, component_state in dynamic_components.items():
        affiliation = name.split(".")
        for idx in range(0, len(affiliation)):
            parent_name = ".".join(affiliation[:-idx])
            has_matched = False
            for flow_name, flow in flow_map.items():
                if flow_name == parent_name:
                    if flow_name not in dynamic_children_state:
                        dynamic_children_state[flow_name] = {}

                    dynamic_children_state[flow_name].update({name.replace(parent_name + ".", ""): component_state})
                    has_matched = True
                    break
            if has_matched:
                break

    # 5: Reload the flow states
    for flow_name, flow in flow_map.items():
        flow.load_state_dict(state.pop(flow_name), dynamic_children_state.get(flow_name, {}), strict=strict)

    # 6: Verify all dynamic components has been re-created.
    if strict:
        components_names = (
            [root_flow.name] + [f.name for f in root_flow.flows.values()] + [w.name for w in root_flow.works()]
        )
        for component_name in dynamic_components:
            if component_name not in components_names:
                raise Exception(f"The component {component_name} was re-created during state reloading.")


class _MagicMockJsonSerializable(MagicMock):
    @staticmethod
    def __json__():
        return "{}"


def _mock_import(*args, original_fn=None):
    try:
        return original_fn(*args)
    except Exception:
        return _MagicMockJsonSerializable()


@contextmanager
def _mock_missing_imports():
    original_fn = builtins.__import__
    builtins.__import__ = functools.partial(_mock_import, original_fn=original_fn)
    try:
        yield
    finally:
        builtins.__import__ = original_fn


def is_static_method(klass_or_instance, attr) -> bool:
    return isinstance(inspect.getattr_static(klass_or_instance, attr), staticmethod)


def _lightning_dispatched() -> bool:
    return bool(int(os.getenv("LIGHTNING_DISPATCHED", 0)))


def _using_debugger() -> bool:
    """This method is used to detect whether the app is run with a debugger attached."""
    if "LIGHTNING_DETECTED_DEBUGGER" in os.environ:
        return True

    # Collect the information about the process.
    parent_process = os.popen(f"ps -ax | grep -i {os.getpid()} | grep -v grep").read()

    # Detect whether VSCode or PyCharm debugger are used
    use_debugger = "debugpy" in parent_process or "pydev" in parent_process

    # Store the result to avoid multiple popen calls.
    if use_debugger:
        os.environ["LIGHTNING_DETECTED_DEBUGGER"] = "1"
    return use_debugger


def _should_dispatch_app() -> bool:
    return (
        not _lightning_dispatched()
        and "LIGHTNING_APP_STATE_URL" not in os.environ
        # Keep last to avoid running it if already dispatched
        and _using_debugger()
    )


def _is_headless(app: "LightningApp") -> bool:
    """Utility which returns True if the given App has no ``Frontend`` objects or URLs exposed through
    ``configure_layout``."""
    if app.frontends:
        return False
    for component in breadth_first(app.root, types=(lightning_app.LightningFlow,)):
        for entry in component._layout:
            if "target" in entry:
                return False
    return True


def _handle_is_headless(app: "LightningApp"):
    """Utility for runtime-specific handling of changes to the ``is_headless`` property."""
    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)

    if app_id is None or project_id is None:
        return

    from lightning_app.utilities.network import LightningClient

    client = LightningClient()
    list_apps_response = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)

    current_lightningapp_instance: Optional[Externalv1LightningappInstance] = None
    for lightningapp_instance in list_apps_response.lightningapps:
        if lightningapp_instance.id == app_id:
            current_lightningapp_instance = lightningapp_instance
            break

    if not current_lightningapp_instance:
        return

    if any(
        [
            current_lightningapp_instance.spec.is_headless == app.is_headless,
            current_lightningapp_instance.status.phase != V1LightningappInstanceState.RUNNING,
        ]
    ):
        return

    current_lightningapp_instance.spec.is_headless = app.is_headless

    client.lightningapp_instance_service_update_lightningapp_instance(
        project_id=project_id,
        id=current_lightningapp_instance.id,
        body=AppinstancesIdBody(name=current_lightningapp_instance.name, spec=current_lightningapp_instance.spec),
    )
